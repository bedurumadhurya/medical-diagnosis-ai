from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def _ensure_log() -> Path:
    path = settings.audit_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_fingerprint(data: bytes, filename: str) -> str:
    digest = hashlib.sha256(data).hexdigest()
    name_hash = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:12]
    return f"{name_hash}:{digest[:16]}"


def write_audit(event: str, payload: dict) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    path = _ensure_log()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def recent_audit(limit: int = 50) -> list[dict]:
    path = settings.audit_log_path
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(rows))
