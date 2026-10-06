"""Identifying a LoRA file across the tabs: its content hash (the Profiler writes it into a report's sidecar) and the
sidecar lookup Repair Studio uses to show a profiled LoRA's report inline."""
import hashlib
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def compute_lora_hash(lora_path: str) -> Optional[str]:
    """SHA-256 of the file bytes — a stable LoRA identifier for cross-tab lookups (the Profiler writes it into its
    sidecar; Repair Studio hashes the loaded primary and scans the profiles folder for a match). None on any IO error
    or a missing path. A ~50 MB LoRA hashes in well under a second."""
    if not lora_path or not os.path.isfile(lora_path):
        return None
    h = hashlib.sha256()
    try:
        with open(lora_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except Exception:
        logger.exception("Failed to compute LoRA hash for %s", lora_path)
        return None
    return h.hexdigest()


def find_profile_for_hash(profiles_dir: str, target_hash: str) -> Optional[dict]:
    """The first *.json sidecar in `profiles_dir` whose `hash` matches `target_hash` (with `_sidecar_path` added), or
    None (no folder, no match, a parse error). Cheap enough for a GUI event handler: one small JSON per profile."""
    if not target_hash or not profiles_dir or not os.path.isdir(profiles_dir):
        return None
    try:
        candidates = sorted(fn for fn in os.listdir(profiles_dir) if fn.endswith(".json"))
    except Exception:
        return None
    for fn in candidates:
        path = os.path.join(profiles_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("hash") == target_hash:
            payload["_sidecar_path"] = path
            return payload
    return None
