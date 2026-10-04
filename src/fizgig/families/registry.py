"""The families described through FamilyDescription - every family Fizgig trains. Their order is the Base Model
dropdown's and the workbench tabs' radio order.
"""
from typing import Optional

from fizgig.families.description import FamilyDescription
from fizgig.families.klein import KLEIN
from fizgig.families.krea2 import KREA2
from fizgig.families.minimax import MINIMAX
from fizgig.families.qwen_image import QWEN_IMAGE_21

FAMILIES = {d.key: d for d in (KLEIN, MINIMAX, KREA2, QWEN_IMAGE_21)}

for _d in FAMILIES.values():
    _problems = _d.validate()
    if _problems:
        raise ValueError(f"family description {_d.key!r} is inconsistent: {'; '.join(_problems)}")


def get(key: str) -> Optional[FamilyDescription]:
    return FAMILIES.get(key)


def by_gui_label(label: str) -> Optional[FamilyDescription]:
    """The description behind a Base Model selector entry, or None."""
    for d in FAMILIES.values():
        if label == d.gui_label or label in d.aliases:
            return d
    return None


def by_arch_id(arch_id: str) -> Optional[FamilyDescription]:
    """The description whose architecture id (cache filenames, metadata) is arch_id, or None.
    Shared code (metadata, dataset buckets) asks this instead of carrying per-family entries."""
    for d in FAMILIES.values():
        if d.arch_id == arch_id:
            return d
    return None


def training_families() -> list:
    """Descriptions whose training entry points exist (shown in the Base Model selector)."""
    return [d for d in FAMILIES.values() if d.training_ready and not d.hidden]


def workbench_families(tool: str) -> list:
    """Descriptions whose driver is built and whose description enables this workbench tool ("repair", ...)."""
    return [d for d in FAMILIES.values() if d.training_ready and not d.hidden and tool in d.workbench]


def family_of_lora(path: str) -> Optional[FamilyDescription]:
    """The described family a LoRA file was written for, from its header alone (key names, no tensor data), or
    None. A file matches when most of its down weights name modules inside the family's block map in the family's
    own key format."""
    try:
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            keys = list(f.keys())
    except Exception:
        return None
    for d in training_families():
        from fizgig.families.lorafile import lokr_modules
        mods = [m for m in (d.lora.module_of(k) for k in keys) if m is not None]
        mods += [m for m, _ in lokr_modules(d, keys) if m is not None and m.startswith(d.block_prefix + ".")]
        if not mods:
            continue
        drv = d.load_driver()
        inside = sum(1 for m in mods if drv.block_of(m) is not None)
        if inside and inside >= 0.5 * len(mods):
            return d
    return None
