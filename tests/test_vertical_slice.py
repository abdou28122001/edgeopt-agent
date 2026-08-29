from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from pathlib import Path

import pytest

from edgeopt.contracts import DeploymentContract, QualityRule, evidence_hash
from edgeopt.demo.prepare import create_fixture
from edgeopt.mcp_server import edgeopt_verify
from edgeopt.runtime import package, run_demo, verify
from edgeopt.trust import attest, ensure_key, verify_attestation


def prepared(tmp_path: Path) -> tuple[Path, dict, dict]:
    spec_path = create_fixture(tmp_path / "input")
    manifest = run_demo(spec_path, tmp_path / "run")
    return spec_path, manifest, json.loads(spec_path.read_text())


def rehash(record: dict) -> dict:
    result = dict(record)
    result["evidence_sha256"] = evidence_hash(result)
    return result


def test_fixture_hash_is_deterministic(tmp_path: Path) -> None:
    first = create_fixture(tmp_path / "one")
    second = create_fixture(tmp_path / "two")
    first_data, second_data = json.loads(first.read_text()), json.loads(second.read_text())
    assert Path(first_data["model_path"]).read_bytes() == Path(second_data["model_path"]).read_bytes()
    assert first_data["contract"]["model"]["artifact_sha256"] == second_data["contract"]["model"]["artifact_sha256"]


def test_runtime_paths_and_content_are_bound_before_inference(tmp_path: Path) -> None:
    spec, _, data = prepared(tmp_path)
    tampered_model = tmp_path / "other.onnx"
    shutil.copy2(data["model_path"], tampered_model)
    altered = dict(data, model_path=str(tampered_model))
    altered_spec = tmp_path / "tampered-path.json"
    altered_spec.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="paths"):
        run_demo(altered_spec, tmp_path / "path-run")
    Path(data["model_path"]).write_bytes(Path(data["model_path"]).read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        run_demo(spec, tmp_path / "content-run")


def test_fixture_content_tamper_is_rejected(tmp_path: Path) -> None:
    spec, _, data = prepared(tmp_path)
    fixture = Path(data["fixture_path"])
    fixture.write_bytes(fixture.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        run_demo(spec, tmp_path / "fixture-run")


def test_evidence_hash_tampering_fails_closed(tmp_path: Path) -> None:
    _, manifest, _ = prepared(tmp_path)
    contract, rule = DeploymentContract(**manifest["contract"]), QualityRule(**manifest["quality_rule"])
    baseline = manifest["baseline"]
    assert verify(contract, baseline, dict(manifest["candidate"], p95_latency_ms=0.0), rule)["decision"] == "rejected"
    assert verify(contract, baseline, rehash(dict(manifest["candidate"], similarity_mae_vs_baseline=99.0)), rule)["decision"] == "rejected"
    assert verify(contract, baseline, rehash(dict(manifest["candidate"], model_sha256="wrong", artifact_size_bytes=1)), rule)["decision"] == "rejected"


def test_cpu_runner_refuses_fake_jetson_target(tmp_path: Path) -> None:
    spec, _, data = prepared(tmp_path)
    changed = dict(data, contract=dict(data["contract"], target={**data["contract"]["target"], "profile_id": "jetson-orin"}))
    bad_spec = tmp_path / "jetson.json"
    bad_spec.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="local demo"):
        run_demo(bad_spec, tmp_path / "jetson-run")


@pytest.mark.parametrize("objective, value", [("max_p95_latency_ms", 0.0000001), ("min_throughput", 10**12), ("max_model_size_mb", 0.0000001)])
def test_hard_constraints_are_reported_and_enforced(tmp_path: Path, objective: str, value: float) -> None:
    _, _, data = prepared(tmp_path)
    changed = dict(data, contract=dict(data["contract"], objectives={**data["contract"]["objectives"], objective: value}))
    spec = tmp_path / f"{objective}.json"
    spec.write_text(json.dumps(changed))
    result = run_demo(spec, tmp_path / objective)
    assert result["decision"]["decision"] == "rejected"
    assert any(item["constraint"] == objective and not item["passed"] for item in result["decision"]["constraints"])


def test_required_memory_constraint_fails_closed(tmp_path: Path) -> None:
    _, _, data = prepared(tmp_path)
    data["contract"]["objectives"]["max_memory_mb"] = 1
    spec = tmp_path / "memory.json"
    spec.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="max_memory_mb"):
        run_demo(spec, tmp_path / "memory-run")


def test_quality_limit_beats_faster_candidate(tmp_path: Path) -> None:
    _, manifest, _ = prepared(tmp_path)
    contract, rule = DeploymentContract(**manifest["contract"]), QualityRule(**manifest["quality_rule"])
    bad_quality = rehash(dict(manifest["candidate"], similarity_mae_vs_baseline=1.0, p95_latency_ms=0.0))
    assert verify(contract, manifest["baseline"], bad_quality, rule)["decision"] == "rejected"


def test_mcp_verification_does_not_trust_stored_decision(tmp_path: Path) -> None:
    _, manifest, _ = prepared(tmp_path)
    path = tmp_path / "manifest.json"
    manifest["decision"] = {"decision": "recommend_candidate", "measured": True, "reason": "fabricated"}
    path.write_text(json.dumps(manifest))
    result = edgeopt_verify(str(path))
    assert result["reason"] != "fabricated"
    assert result["decision"] in {"recommend_candidate", "recommend_baseline"}


def test_package_requires_verified_approval_and_artifact_identity(tmp_path: Path) -> None:
    _, manifest, data = prepared(tmp_path)
    model = Path(data["model_path"])
    with pytest.raises(PermissionError, match="approved"):
        package(model, tmp_path / "no-approval", manifest, approved=False)
    fabricated = dict(manifest, decision={"decision": "recommend_candidate"})
    with pytest.raises((KeyError, PermissionError, ValueError)):
        package(model, tmp_path / "fabricated", fabricated, approved=True)
    unrelated = tmp_path / "unrelated.onnx"
    unrelated.write_bytes(model.read_bytes())
    with pytest.raises(PermissionError, match="verified"):
        package(unrelated, tmp_path / "unrelated-package", manifest, approved=True)
    tampered = tmp_path / "tampered.onnx"
    shutil.copy2(model, tampered)
    tampered.write_bytes(tampered.read_bytes() + b"changed")
    with pytest.raises(PermissionError, match="verified"):
        package(tampered, tmp_path / "tampered-package", manifest, approved=True)
    rejected = dict(manifest, candidate=rehash(dict(manifest["candidate"], similarity_mae_vs_baseline=9.0)))
    with pytest.raises(PermissionError, match="recommendation"):
        package(model, tmp_path / "rejected-package", rejected, approved=True)
    output = package(model, tmp_path / "valid-package", manifest, approved=True)
    assert (output / "run-manifest.json").exists()
    assert (output / model.name).exists()


def test_recomputed_public_hash_without_trusted_tag_still_fails(tmp_path: Path) -> None:
    _, manifest, _ = prepared(tmp_path)
    contract, rule = DeploymentContract(**manifest["contract"]), QualityRule(**manifest["quality_rule"])
    forged = dict(manifest["candidate"], p95_latency_ms=0.0)
    forged["evidence_sha256"] = evidence_hash(forged)
    assert verify(contract, manifest["baseline"], forged, rule)["decision"] == "rejected"


def test_replaced_attestation_key_cannot_validate_old_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_state = tmp_path / "state-a"
    monkeypatch.setenv("EDGEOPT_STATE_DIR", str(first_state))
    _, manifest, _ = prepared(tmp_path / "run-a")
    contract, rule = DeploymentContract(**manifest["contract"]), QualityRule(**manifest["quality_rule"])
    monkeypatch.setenv("EDGEOPT_STATE_DIR", str(tmp_path / "state-b"))
    result = verify(contract, manifest["baseline"], manifest["candidate"], rule)
    assert result["decision"] == "rejected"


def test_rule_rebinding_fails_closed(tmp_path: Path) -> None:
    _, manifest, _ = prepared(tmp_path)
    contract = DeploymentContract(**manifest["contract"])
    changed_rule = QualityRule(**dict(manifest["quality_rule"], allowed_degradation=99.0))
    assert verify(contract, manifest["baseline"], manifest["candidate"], changed_rule)["decision"] == "rejected"


def test_binding_helper_checks_rule_baseline_directly(tmp_path: Path) -> None:
    _, manifest, _ = prepared(tmp_path)
    from edgeopt.contracts import ensure_same_binding
    rule = QualityRule(**manifest["quality_rule"])
    substituted = QualityRule(**dict(manifest["quality_rule"], baseline_evidence_id="other-baseline"))
    with pytest.raises(ValueError, match="quality rule baseline"):
        ensure_same_binding(manifest["candidate"], manifest["contract"]["evaluation"]["evaluation_id"], substituted, manifest["baseline"])
    with pytest.raises(ValueError, match="quality rule baseline"):
        ensure_same_binding(manifest["candidate"], manifest["contract"]["evaluation"]["evaluation_id"], rule, dict(manifest["baseline"], evidence_sha256="stale"))


def test_package_rejects_contract_artifact_hash_mismatch(tmp_path: Path) -> None:
    _, manifest, data = prepared(tmp_path)
    manifest["contract"]["model"]["artifact_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="verified"):
        package(Path(data["model_path"]), tmp_path / "hash-mismatch", manifest, approved=True)


@pytest.mark.parametrize("key, value", [
    ("max_p95_latency_ms", "fast"),
    ("min_throughput", float("nan")),
    ("max_model_size_mb", float("inf")),
    ("max_p95_latency_ms", -1),
    ("min_throughput", True),
])
def test_malformed_numeric_objectives_fail_controlled(tmp_path: Path, key: str, value: object) -> None:
    _, _, data = prepared(tmp_path)
    changed = dict(data["contract"], objectives={**data["contract"]["objectives"], key: value})
    with pytest.raises(ValueError, match="finite non-negative"):
        DeploymentContract(**changed).validate()


def _concurrent_key_worker(state: str, queue: object) -> None:
    os.environ["EDGEOPT_STATE_DIR"] = state
    from edgeopt.trust import ensure_key
    queue.put(ensure_key().hex())


def test_first_use_key_creation_is_cross_process_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "concurrent-state"
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=_concurrent_key_worker, args=(str(state), queue)) for _ in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    values = [queue.get(timeout=2) for _ in processes]
    assert len(set(values)) == 1
    assert state.joinpath("attestation.key").read_bytes().hex() == values[0]
    monkeypatch.setenv("EDGEOPT_STATE_DIR", str(state))
    record = {"measurement": 1, "attestation_key_id": ""}
    record["attestation_key_id"], record["attestation_tag"] = attest(record)
    assert verify_attestation(record)
