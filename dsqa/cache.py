from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dsqa.config import CONFIG


def _path(kind: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    folder = Path(CONFIG.cache_dir) / kind / digest[:2]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.json"


def get_json(kind: str, key: str) -> object | None:
    path = _path(kind, key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def put_json(kind: str, key: str, value: object) -> None:
    path = _path(kind, key)
    path.write_text(json.dumps(value), encoding="utf-8")
