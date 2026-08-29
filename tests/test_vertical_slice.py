from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeopt.contracts import DeploymentContract, canonical_hash, ensure_same_binding
from edgeopt.demo.prepare import create_fixture
from edgeopt.runtime import package, run_demo


def test_fixture_is_deterministic(tmp_path: Path) -> None:
    first = create_fixture(tmp_path / "one")
    second = create_fixture(tmp_path / "two")
    assert Path(json.loads(first.read_text())["model_path"]).read_bytes() == Path(json.loads(second.read_text())["model_path"]).read_bytes()
    assert json.loads(first.read_text())["contract"]["model"]["artifact_sha256"] == json.loads(second.read_text())["contract"]["model"]["artifact_sha256"]


def test_end_to_end_and_rejection(tmp_path: Path) -> None:
    spec = create_fixture(tmp_path / "input")
    output = tmp_path / "run"
    manifest = run_demo(spec, output)
    assert manifest["baseline"]["p95_latency_ms"] >= 0
    assert manifest["candidate"]["status"] == "measured"
    assert manifest["unsupported_candidate"]["executed"] is False
    assert manifest["decision"]["measured"] is True
    assert not (output / "package").exists()


def test_binding_fails_closed(tmp_path: Path) -> None:
    spec = create_fixture(tmp_path / "input")
    manifest = run_demo(spec, tmp_path / "run")
    from edgeopt.contracts import QualityRule
    rule = QualityRule(**manifest["quality_rule"])
    bad = dict(manifest["candidate"], evaluation_id="stale")
    result = __import__("edgeopt.runtime", fromlist=["verify"]).verify(DeploymentContract(**manifest["contract"]), manifest["baseline"], bad, rule)
    assert result["decision"] == "rejected"


def test_package_gate(tmp_path: Path) -> None:
    spec = create_fixture(tmp_path / "input")
    manifest = run_demo(spec, tmp_path / "run")
    model = Path(json.loads(spec.read_text())["model_path"])
    with pytest.raises(PermissionError):
        package(model, tmp_path / "package", manifest, approved=False)
    assert package(model, tmp_path / "package", manifest, approved=True).exists()
    assert (tmp_path / "package" / "run-manifest.json").exists()
