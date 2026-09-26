"""The families described through FamilyDescription.

Only new families register here. Klein, Krea 2 and MiniMax H3 keep their existing code paths and are
deliberately absent: the GUI's generic hooks return None for them, so every existing branch runs as before.
"""
from typing import Optional

from fizgig.families.description import FamilyDescription
from fizgig.families.qwen_image import QWEN_IMAGE_21

FAMILIES = {d.key: d for d in (QWEN_IMAGE_21,)}

for _d in FAMILIES.values():
    _problems = _d.validate()
    if _problems:
        raise ValueError(f"family description {_d.key!r} is inconsistent: {'; '.join(_problems)}")


def get(key: str) -> Optional[FamilyDescription]:
    return FAMILIES.get(key)


def by_gui_label(label: str) -> Optional[FamilyDescription]:
    """The description behind a Base Model selector entry, or None for Klein / Krea 2 / H3."""
    for d in FAMILIES.values():
        if label == d.gui_label or label in d.aliases:
            return d
    return None


def by_arch_id(arch_id: str) -> Optional[FamilyDescription]:
    """The description whose architecture id (cache filenames, metadata) is arch_id; None for the old families.
    Shared code (metadata, dataset buckets) asks this instead of carrying per-family entries."""
    for d in FAMILIES.values():
        if d.arch_id == arch_id:
            return d
    return None


def training_families() -> list:
    """Descriptions whose training entry points exist (shown in the Base Model selector)."""
    return [d for d in FAMILIES.values() if d.training_ready]
