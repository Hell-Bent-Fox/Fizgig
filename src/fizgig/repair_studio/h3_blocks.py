"""MiniMax H3 block layout for the Repair Studio (and Explorer / Royale in H3 mode).
Companion to `state.py`'s Klein layout — the `SliderState` dataclass
is shared and model-agnostic; only the block-id set + the per-block regex differ.

An H3 full-model LoRA covers (from real trained LoRAs):
- **50 main transformer blocks** — `lora_unet_blocks_0..49_*` (attn qkv/out + mlp fc1/fc2,
  plus `adaln_proj_linear` on pruned-checkpoint runs)
- **2 token-refiner blocks** — `lora_unet_token_refiner_blocks_0/1_*`

So there are **52 per-block sliders**. The block-id namespace is `h3blk_N` / `h3_rf_N` —
deliberately DISTINCT from Krea 2's `block_N` even though both families' LoRA keys start
`lora_unet_blocks_`: bake.py maps block ids across families with one table and relies on the
id namespaces never colliding.

Like Krea 2 there is **no semantic bucket map yet** (nobody has published which of H3's 50
blocks carry identity, style or motion) — these are generic per-block controls, which is
exactly the instrument for *discovering* H3's block semantics.
"""

import re
from typing import List, Set

H3_MAIN_BLOCK_COUNT = 50
H3_REFINER_IDS = ["h3_rf_0", "h3_rf_1"]

# For extracting block ids back out of LoRA module names. Refiner first: its keys
# ("lora_unet_token_refiner_blocks_N_") don't contain the literal "lora_unet_blocks_",
# but testing the specific pattern first keeps that from being load-bearing.
_RF_RX = re.compile(r"lora_unet_token_refiner_blocks_(\d+)_")
_MAIN_RX = re.compile(r"lora_unet_blocks_(\d+)_")


def all_block_ids_h3() -> List[str]:
    """The 52 per-block slider ids: h3blk_0..h3blk_49, then the two token-refiner blocks."""
    return [f"h3blk_{i}" for i in range(H3_MAIN_BLOCK_COUNT)] + list(H3_REFINER_IDS)


def block_regex_h3(block_id: str) -> str:
    """Regex matching a LoRA module's `lora_name` for this block (used by
    `set_module_multiplier_by_pattern`, which does `re.search` on the name).

    Main blocks anchor on `lora_unet_blocks_<N>_` — the `lora_unet_` prefix excludes the
    token-refiner blocks, and the trailing `_` stops `blocks_2_` matching `blocks_20_`.
    """
    if block_id.startswith("h3_rf_"):
        idx = block_id.rsplit("_", 1)[1]
        return rf"token_refiner_blocks_{idx}_"
    if block_id.startswith("h3blk_"):
        idx = block_id.split("_", 1)[1]
        return rf"lora_unet_blocks_{idx}_"
    raise ValueError(f"unknown H3 block id: {block_id!r}")


def extract_block_ids_h3(network) -> Set[str]:
    """Return the set of block ids that have LoRA modules in this network (so the UI can
    grey out sliders for blocks a given LoRA doesn't touch)."""
    ids: Set[str] = set()
    if network is None:
        return ids
    for mod in getattr(network, "unet_loras", []):
        name = mod.lora_name
        r = _RF_RX.search(name)
        if r:
            ids.add(f"h3_rf_{int(r.group(1))}")
            continue
        m = _MAIN_RX.search(name)
        if m:
            ids.add(f"h3blk_{int(m.group(1))}")
    return ids
