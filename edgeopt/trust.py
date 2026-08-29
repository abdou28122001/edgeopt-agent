"""Small local authenticity boundary for EdgeOpt evidence and rules."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any


def _state_dir() -> Path:
    return Path(os.environ.get("EDGEOPT_STATE_DIR", ".edgeopt-state"))


def _key_path() -> Path:
    return _state_dir() / "attestation.key"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def ensure_key() -> bytes:
    path = _key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        key = path.read_bytes()
        if len(key) < 32:
            raise ValueError("trusted attestation key is invalid")
        return key
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def key_id(key: bytes | None = None) -> str:
    return hashlib.sha256(key or ensure_key()).hexdigest()[:16]


def _payload(record: dict[str, Any]) -> bytes:
    return _canonical({k: v for k, v in record.items() if k != "attestation_tag"})


def attest(record: dict[str, Any]) -> tuple[str, str]:
    key = ensure_key()
    tag = hmac.new(key, _payload(record), hashlib.sha256).hexdigest()
    return key_id(key), tag


def verify_attestation(record: dict[str, Any]) -> bool:
    try:
        key = ensure_key()
        if not hmac.compare_digest(record.get("attestation_key_id", ""), key_id(key)):
            return False
        expected = hmac.new(key, _payload(record), hashlib.sha256).hexdigest()
        return hmac.compare_digest(record.get("attestation_tag", ""), expected)
    except (OSError, ValueError, TypeError):
        return False
