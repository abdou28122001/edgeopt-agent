"""Typed, serializable contracts used by the deterministic vertical slice."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def evidence_hash(record: dict[str, Any]) -> str:
    """Hash all persisted evidence except the self-referential hash field."""
    return canonical_hash({key: value for key, value in record.items() if key != "evidence_sha256"})


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
            raise ValueError("max_memory_mb is not measured by the CPU v0 path")

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_quality_rule(evaluation_id: str, baseline_id: str, baseline_hash: str) -> QualityRule:
    body = {
        "evaluation_id": evaluation_id,
        "metric_id": "synthetic_output_mae_vs_baseline",
        "direction": "lower_is_better",
        "drop_mode": "absolute",
        "allowed_degradation": 1e-6,
        "baseline_evidence_id": baseline_id,
        "baseline_evidence_sha256": baseline_hash,
        "version": "v0",
    }
    return QualityRule(quality_rule_id=canonical_hash(body)[:16], **{k: body[k] for k in body if k != "quality_rule_id"})


def ensure_same_binding(candidate: dict[str, Any], evaluation_id: str, rule: QualityRule, baseline: dict[str, Any]) -> None:
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
