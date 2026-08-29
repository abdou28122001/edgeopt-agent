"""CPU ONNX Runtime execution, measurement, verification, and packaging."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .contracts import DeploymentContract, QualityRule, canonical_hash, file_sha256, ensure_same_binding, make_quality_rule


def _session(model: Path, optimized: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])


def measure(model: Path, inputs: dict[str, np.ndarray], optimized: bool, warmups: int = 3, repeats: int = 15) -> tuple[list[np.ndarray], dict[str, Any]]:
    session = _session(model, optimized)
    for _ in range(warmups):
        session.run(None, inputs)
    samples: list[float] = []
    outputs: list[np.ndarray] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = session.run(None, inputs)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
        outputs = [np.asarray(result[0])]
    return outputs, {
        "p50_latency_ms": float(np.percentile(samples, 50)),
        "p95_latency_ms": float(np.percentile(samples, 95)),
        "throughput_per_second": float(1000 / np.mean(samples)),
        "warmups": warmups,
        "repeats": repeats,
        "runtime": "onnxruntime",
        "runtime_version": ort.__version__,
        "provider": "CPUExecutionProvider",
    }


def _evidence(run_id: str, candidate_id: str, contract: DeploymentContract, model: Path, fixture: Path, output: np.ndarray, metrics: dict[str, Any], baseline: dict[str, Any] | None, rule: QualityRule | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": run_id,
        "evidence_id": f"{run_id}-{candidate_id}",
        "candidate_id": candidate_id,
        "model_sha256": file_sha256(model),
        "artifact_size_bytes": model.stat().st_size,
        "evaluation_id": contract.evaluation["evaluation_id"],
        "fixture_ref": str(fixture),
        "fixture_sha256": file_sha256(fixture),
        "preprocessing_id": contract.evaluation["preprocessing"]["id"],
        "preprocessing_sha256": contract.evaluation["preprocessing"]["sha256"],
        "device_profile_id": contract.target["profile_id"],
        "provider": metrics["provider"],
        "runtime": metrics["runtime"],
        "runtime_version": metrics["runtime_version"],
        "warmups": metrics["warmups"],
        "repeats": metrics["repeats"],
        "p50_latency_ms": metrics["p50_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "throughput_per_second": metrics["throughput_per_second"],
        "similarity_mae_vs_baseline": None,
        "status": "measured",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if baseline is not None and rule is not None:
        record["similarity_mae_vs_baseline"] = float(np.mean(np.abs(output - np.asarray(baseline["output"]))))
        record.update({
            "quality_rule_id": rule.quality_rule_id,
            "quality_rule_version": rule.version,
            "quality_rule_sha256": canonical_hash(rule.to_dict()),
            "baseline_evidence_id": baseline["evidence_id"],
            "baseline_evidence_sha256": baseline["evidence_sha256"],
        })
    record["evidence_sha256"] = canonical_hash({k: v for k, v in record.items() if k != "evidence_sha256"})
    return record


def verify(contract: DeploymentContract, baseline: dict[str, Any], candidate: dict[str, Any], rule: QualityRule) -> dict[str, Any]:
    contract.validate()
    try:
        ensure_same_binding(candidate, contract.evaluation["evaluation_id"], rule, baseline)
    except ValueError as exc:
        return {"decision": "rejected", "reason": str(exc), "measured": False}
    mae = candidate.get("similarity_mae_vs_baseline")
    if not isinstance(mae, (int, float)) or not np.isfinite(mae):
        return {"decision": "rejected", "reason": "missing or non-finite similarity evidence", "measured": False}
    if mae > rule.allowed_degradation:
        return {"decision": "rejected", "reason": "candidate exceeds frozen quality rule", "measured": True, "similarity_mae": mae}
    decision = "recommend_candidate" if candidate["p95_latency_ms"] < baseline["p95_latency_ms"] else "recommend_baseline"
    return {"decision": decision, "reason": "measured p95 comparison under frozen rule", "measured": True, "similarity_mae": mae, "baseline_p95_latency_ms": baseline["p95_latency_ms"], "candidate_p95_latency_ms": candidate["p95_latency_ms"]}


def reject_unsupported(contract: DeploymentContract) -> dict[str, Any]:
    return {"candidate_id": "candidate-b-tensorrt", "status": "rejected", "executed": False, "reason": "requires CUDA/TensorRT; CPU demo profile provides only CPUExecutionProvider"}


def package(selected_model: Path, output: Path, manifest: dict[str, Any], approved: bool) -> Path:
    if not approved:
        raise PermissionError("package requires explicit approved=true")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_model, output / selected_model.name)
    (output / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text("# EdgeOpt package\n\nSelected artifact was packaged after explicit approval.\n")
    return output


def run_demo(spec_path: Path, output: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    contract = DeploymentContract(**spec["contract"])
    contract.validate()
    model = Path(spec["model_path"])
    fixture = Path(spec["fixture_path"])
    inputs = {"input": np.load(fixture)["input"].astype(np.float32)}
    run_id = spec.get("run_id", "edgeopt-local")
    base_output, base_metrics = measure(model, inputs, optimized=False)
    baseline = _evidence(run_id, "baseline", contract, model, fixture, base_output[0], base_metrics, None, None)
    baseline["output"] = base_output[0].tolist()
    baseline["evidence_sha256"] = canonical_hash({k: v for k, v in baseline.items() if k not in {"evidence_sha256", "output"}})
    rule = make_quality_rule(contract.evaluation["evaluation_id"], baseline["evidence_id"], baseline["evidence_sha256"])
    cand_output, cand_metrics = measure(model, inputs, optimized=True)
    candidate = _evidence(run_id, "candidate-a-ort-optimized", contract, model, fixture, cand_output[0], cand_metrics, baseline, rule)
    decision = verify(contract, baseline, candidate, rule)
    unsupported = reject_unsupported(contract)
    output.mkdir(parents=True, exist_ok=True)
    baseline.pop("output", None)
    for name, value in [("baseline.json", baseline), ("candidate-a.json", candidate), ("candidate-b.json", unsupported), ("decision.json", decision), ("quality-rule.json", rule.to_dict())]:
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    manifest = {"run_id": run_id, "contract": contract.to_dict(), "baseline": baseline, "candidate": candidate, "unsupported_candidate": unsupported, "quality_rule": rule.to_dict(), "decision": decision, "package": {"approved": False, "produced": False}, "measured_vs_assumed": {"measurements": "local CPU ONNX Runtime", "target": "profiled CPU only"}}
    (output / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
