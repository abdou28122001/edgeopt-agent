"""Typed, serializable contracts used by the deterministic vertical slice."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .trust import attest, ensure_key, key_id, verify_attestation


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def evidence_hash(record: dict[str, Any]) -> str:
    """Hash content, excluding self-hash and derived HMAC tag only."""
    return canonical_hash({key: value for key, value in record.items() if key not in {"evidence_sha256", "attestation_tag"}})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeploymentContract:
    model: dict[str, Any]
    evaluation: dict[str, Any]
    target: dict[str, Any]
    application_context: dict[str, Any]
    objectives: dict[str, Any]

    def validate(self) -> None:
        for section in ("model", "evaluation", "target", "application_context", "objectives"):
            value = getattr(self, section)
            if not isinstance(value, dict) or not value:
                raise ValueError(f"deployment contract section is missing: {section}")
        if self.model.get("format") != "onnx":
            raise ValueError("T006 requires an ONNX model contract")
        if self.objectives.get("priority") not in {"speed", "quality", "memory", "balanced", "custom"}:
            raise ValueError("objectives.priority is not a supported value")
        if self.objectives.get("priority") == "custom" and not self.objectives.get("custom_metric"):
            raise ValueError("custom priority requires custom_metric")
        if self.objectives.get("max_memory_mb") is not None:
            raise ValueError("max_memory_mb is unsupported and not measured by the CPU v0 path")
        for key in ("max_quality_degradation", "max_p95_latency_ms", "min_throughput", "max_model_size_mb"):
            value = self.objectives.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"objectives.{key} must be a finite non-negative number")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class QualityRule:
    quality_rule_id: str
    version: str
    evaluation_id: str
    metric_id: str
    direction: str
    drop_mode: str
    allowed_degradation: float
    baseline_evidence_id: str
    baseline_evidence_sha256: str
    baseline_attestation_key_id: str
    baseline_attestation_tag: str
    attestation_key_id: str
    attestation_tag: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_quality_rule(evaluation_id: str, baseline: dict[str, Any]) -> QualityRule:
    body = {
        "evaluation_id": evaluation_id,
        "metric_id": "synthetic_output_mae_vs_baseline",
        "direction": "lower_is_better",
        "drop_mode": "absolute",
        "allowed_degradation": 1e-6,
        "baseline_evidence_id": baseline["evidence_id"],
        "baseline_evidence_sha256": evidence_hash(baseline),
        "version": "v0",
        "baseline_attestation_key_id": baseline["attestation_key_id"],
        "baseline_attestation_tag": baseline["attestation_tag"],
        "attestation_key_id": key_id(ensure_key()),
    }
    body["quality_rule_id"] = canonical_hash({k: v for k, v in body.items() if k != "attestation_tag"})[:16]
    _, tag = attest(body)
    body["attestation_tag"] = tag
    return QualityRule(**body)


def ensure_same_binding(candidate: dict[str, Any], evaluation_id: str, rule: QualityRule, baseline: dict[str, Any]) -> None:
    if rule.baseline_evidence_id != baseline.get("evidence_id") or rule.baseline_evidence_sha256 != baseline.get("evidence_sha256") or rule.baseline_evidence_sha256 != evidence_hash(baseline):
        raise ValueError("quality rule baseline binding mismatch")
    required = {
        "evaluation_id": evaluation_id,
        "quality_rule_id": rule.quality_rule_id,
        "quality_rule_version": rule.version,
        "quality_rule_sha256": canonical_hash(rule.to_dict()),
        "baseline_evidence_id": baseline["evidence_id"],
        "baseline_evidence_sha256": baseline["evidence_sha256"],
    }
    for key, expected in required.items():
        if candidate.get(key) != expected:
            raise ValueError(f"evidence binding mismatch: {key}")


def authenticate_rule(rule: QualityRule, baseline: dict[str, Any]) -> bool:
    if rule.baseline_evidence_id != baseline.get("evidence_id") or rule.baseline_evidence_sha256 != baseline.get("evidence_sha256") or rule.baseline_evidence_sha256 != evidence_hash(baseline):
        return False
    if rule.baseline_attestation_key_id != baseline.get("attestation_key_id") or rule.baseline_attestation_tag != baseline.get("attestation_tag"):
        return False
    return verify_attestation(rule.to_dict())
