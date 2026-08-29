"""CPU ONNX Runtime execution, measurement, verification, and packaging."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .contracts import DeploymentContract, QualityRule, authenticate_rule, canonical_hash, evidence_hash, file_sha256, ensure_same_binding, make_quality_rule
from .trust import attest, ensure_key, key_id, verify_attestation


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


def _evidence(run_id: str, candidate_id: str, contract: DeploymentContract, model: Path, fixture: Path, output: np.ndarray, metrics: dict[str, Any], baseline_output: np.ndarray | None, baseline: dict[str, Any] | None, rule: QualityRule | None) -> dict[str, Any]:
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
        "execution_profile_id": "local-cpu-onnxruntime",
        "measurement_source": "local_cpu_execution",
        "attestation_key_id": key_id(ensure_key()),
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
    if baseline_output is not None and baseline is not None and rule is not None:
        record["similarity_mae_vs_baseline"] = float(np.mean(np.abs(output - baseline_output)))
        record.update({
            "quality_rule_id": rule.quality_rule_id,
            "quality_rule_version": rule.version,
            "quality_rule_sha256": canonical_hash(rule.to_dict()),
            "baseline_evidence_id": baseline["evidence_id"],
            "baseline_evidence_sha256": baseline["evidence_sha256"],
        })
    record["evidence_sha256"] = evidence_hash(record)
    record["attestation_key_id"], record["attestation_tag"] = attest(record)
    return record


def _constraint_results(contract: DeploymentContract, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    objectives = contract.objectives
    results: list[dict[str, Any]] = []
    checks = [("max_p95_latency_ms", "p95_latency_ms", "lte"), ("min_throughput", "throughput_per_second", "gte"), ("max_model_size_mb", "artifact_size_bytes", "lte")]
    for objective_key, evidence_key, operator in checks:
        limit = objectives.get(objective_key)
        if limit is None:
            continue
        value = evidence.get(evidence_key)
        if value is None:
            results.append({"constraint": objective_key, "passed": False, "reason": "required measurement missing"})
            continue
        actual = value / (1024 * 1024) if objective_key == "max_model_size_mb" else value
        passed = actual <= limit if operator == "lte" else actual >= limit
        results.append({"constraint": objective_key, "passed": bool(passed), "actual": actual, "limit": limit, "reason": "within limit" if passed else "hard limit violated"})
    return results


def _integrity_ok(evidence: dict[str, Any]) -> bool:
    return isinstance(evidence.get("evidence_sha256"), str) and evidence["evidence_sha256"] == evidence_hash(evidence)


def verify(contract: DeploymentContract, baseline: dict[str, Any], candidate: dict[str, Any], rule: QualityRule) -> dict[str, Any]:
    contract.validate()
    if contract.target.get("profile_id") != "local-cpu-onnxruntime" or contract.target.get("provider") != "cpu" or "CPUExecutionProvider" not in contract.target.get("verified_capabilities", []):
        return {"decision": "rejected", "reason": "local CPU execution requires the local-cpu-onnxruntime target profile", "measured": False, "constraints": []}
    if not _integrity_ok(baseline) or not _integrity_ok(candidate) or not verify_attestation(baseline) or not verify_attestation(candidate):
        return {"decision": "rejected", "reason": "evidence integrity or trusted attestation failed", "measured": False, "constraints": []}
    if not authenticate_rule(rule, baseline):
        return {"decision": "rejected", "reason": "quality rule baseline binding or attestation failed", "measured": False, "constraints": []}
    if candidate.get("model_sha256") != baseline.get("model_sha256") or candidate.get("artifact_size_bytes") != baseline.get("artifact_size_bytes"):
        return {"decision": "rejected", "reason": "candidate artifact identity differs from measured baseline", "measured": False, "constraints": []}
    try:
        ensure_same_binding(candidate, contract.evaluation["evaluation_id"], rule, baseline)
    except ValueError as exc:
        return {"decision": "rejected", "reason": str(exc), "measured": False, "constraints": []}
    mae = candidate.get("similarity_mae_vs_baseline")
    if not isinstance(mae, (int, float)) or not np.isfinite(mae):
        return {"decision": "rejected", "reason": "missing or non-finite similarity evidence", "measured": False, "constraints": []}
    constraints = _constraint_results(contract, candidate)
    baseline_constraints = _constraint_results(contract, baseline)
    if mae > rule.allowed_degradation:
        return {"decision": "rejected", "reason": "candidate exceeds frozen quality rule", "measured": True, "similarity_mae": mae, "constraints": constraints}
    if any(not item["passed"] for item in constraints):
        return {"decision": "rejected", "reason": "candidate violates a hard deployment constraint", "measured": True, "similarity_mae": mae, "constraints": constraints}
    if any(not item["passed"] for item in baseline_constraints):
        return {"decision": "recommend_candidate", "reason": "baseline violates a hard deployment constraint; candidate satisfies it", "measured": True, "similarity_mae": mae, "constraints": constraints, "baseline_constraints": baseline_constraints, "baseline_p95_latency_ms": baseline["p95_latency_ms"], "candidate_p95_latency_ms": candidate["p95_latency_ms"]}
    priority = contract.objectives["priority"]
    if priority == "quality":
        choose_candidate = mae < 1e-12
    elif priority == "memory":
        choose_candidate = candidate["artifact_size_bytes"] < baseline["artifact_size_bytes"]
    else:
        choose_candidate = candidate["p95_latency_ms"] < baseline["p95_latency_ms"]
    return {"decision": "recommend_candidate" if choose_candidate else "recommend_baseline", "reason": "measured comparison under frozen rule and hard constraints", "measured": True, "similarity_mae": mae, "constraints": constraints, "baseline_constraints": baseline_constraints, "baseline_p95_latency_ms": baseline["p95_latency_ms"], "candidate_p95_latency_ms": candidate["p95_latency_ms"]}


def reject_unsupported(contract: DeploymentContract) -> dict[str, Any]:
    return {"candidate_id": "candidate-b-tensorrt", "status": "rejected", "executed": False, "reason": "requires CUDA/TensorRT; CPU demo profile provides only CPUExecutionProvider"}


def package(selected_model: Path, output: Path, manifest: dict[str, Any], approved: bool) -> Path:
    if not approved:
        raise PermissionError("package requires explicit approved=true")
    contract = DeploymentContract(**manifest["contract"])
    rule = QualityRule(**manifest["quality_rule"])
    decision = verify(contract, manifest["baseline"], manifest["candidate"], rule)
    if decision.get("decision") not in {"recommend_candidate", "recommend_baseline"}:
        raise PermissionError("package requires a freshly verified recommendation")
    stored_decision = manifest.get("decision")
    if not isinstance(stored_decision, dict) or stored_decision.get("decision") != decision.get("decision") or stored_decision.get("measured") is not True or stored_decision.get("constraints") != decision.get("constraints"):
        raise PermissionError("manifest decision does not match freshly verified recommendation")
    selected = manifest["candidate"] if decision["decision"] == "recommend_candidate" else manifest["baseline"]
    expected_path = Path(contract.model["source"]).resolve()
    if Path(selected_model).resolve() != expected_path or file_sha256(selected_model) != selected["model_sha256"] or selected["model_sha256"] != contract.model["artifact_sha256"]:
        raise PermissionError("selected artifact is not the verified measured artifact")
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
    declared_model = Path(contract.model["source"])
    declared_fixture = Path(contract.evaluation["fixture_ref"])
    if model.resolve() != declared_model.resolve() or fixture.resolve() != declared_fixture.resolve():
        raise ValueError("runtime paths do not match deployment contract references")
    if not model.is_file() or not fixture.is_file() or file_sha256(model) != contract.model["artifact_sha256"] or file_sha256(fixture) != contract.evaluation["fixture_sha256"]:
        raise ValueError("runtime artifact or fixture hash does not match deployment contract")
    if contract.target.get("profile_id") != "local-cpu-onnxruntime" or contract.target.get("provider") != "cpu" or "CPUExecutionProvider" not in contract.target.get("verified_capabilities", []):
        raise ValueError("local demo refuses a non-local CPU execution target")
    inputs = {"input": np.load(fixture)["input"].astype(np.float32)}
    run_id = spec.get("run_id", "edgeopt-local")
    base_output, base_metrics = measure(model, inputs, optimized=False)
    baseline = _evidence(run_id, "baseline", contract, model, fixture, base_output[0], base_metrics, None, None, None)
    rule = make_quality_rule(contract.evaluation["evaluation_id"], baseline)
    cand_output, cand_metrics = measure(model, inputs, optimized=True)
    candidate = _evidence(run_id, "candidate-a-ort-optimized", contract, model, fixture, cand_output[0], cand_metrics, base_output[0], baseline, rule)
    decision = verify(contract, baseline, candidate, rule)
    unsupported = reject_unsupported(contract)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [("baseline.json", baseline), ("candidate-a.json", candidate), ("candidate-b.json", unsupported), ("decision.json", decision), ("quality-rule.json", rule.to_dict())]:
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    manifest = {"run_id": run_id, "contract": contract.to_dict(), "baseline": baseline, "candidate": candidate, "unsupported_candidate": unsupported, "quality_rule": rule.to_dict(), "decision": decision, "package": {"approved": False, "produced": False}, "measured_vs_assumed": {"measurements": "local CPU ONNX Runtime", "target": "profiled CPU only"}}
    (output / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
