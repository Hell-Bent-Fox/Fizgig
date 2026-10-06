"""H3 ReFLoRAs: one .safetensors holding a MiniMax H3 LoRA and its RefMod references, in the container format of
@malcolmamal's ComfyUI-MiniMaxH3RefLoRA (HYBRID_FORMAT.md, "H3 RefLoRA v1"), which its Load H3 RefLoRA node reads.

  tensors   the LoRA's own (lora_unet_* ...), as stored
            ref_0 .. ref_n   [1, 24, T, H, W] visual / [1, 32, 2, T] audio, as stored in their RefMod files
  header    the LoRA's metadata (ss_*, modelspec.*)
            refmod_meta  a version-5 RefMod bundle (ComfyUI-MiniMaxH3Mod's BUNDLE_FORMAT): its members are the
                         references' own RefMod metadata (version 4)
            h3_hybrid    {"version": 1, "lora": {"format", "keys", "network_dim", "network_alpha"},
                          "refmod_count", "packed_by", "sources": {"lora", "refmods"}}

A plain LoRA loader ignores the ref_* tensors and a RefMod loader ignores the LoRA's, so the file works in either.
"""
import json
import os

from fizgig.minimax.refmod import BUNDLE_FORMAT_VERSION, NODE_FORMAT_VERSION, NODE_META_KEY

HYBRID_KEY = "h3_hybrid"
HYBRID_VERSION = 1
MAX_REFERENCES = 256


def is_reference_key(key: str) -> bool:
    """A RefMod tensor (a ReFLoRA's ref_N, a standalone mod's latent) rather than a LoRA weight."""
    return key == "latent" or (key.startswith("ref_") and key[4:].isdigit())


def lora_tensors(sd: dict) -> dict:
    """A LoRA file's state dict without the references a ReFLoRA carries beside the weights."""
    return {k: v for k, v in sd.items() if not is_reference_key(k)}


def is_reflora(path: str) -> bool:
    """A file with both LoRA weights and RefMod references."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        return False
    return any(is_reference_key(k) for k in keys) and any(not is_reference_key(k) for k in keys)


def _raw_member(path: str, tensor_key: str):
    """(the reference's own RefMod metadata, tensor as stored) for one reference of a RefMod file: a standalone
    mod's `latent` or a bundle's `ref_N`."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = json.loads((f.metadata() or {}).get(NODE_META_KEY) or "{}")
        tensor = f.get_tensor(tensor_key)
    if str(meta.get("kind", "")) == "bundle":
        meta = (meta.get("members") or [])[int(tensor_key.split("_")[1])]
    return dict(meta), tensor


def save_reflora(lora_path: str, references, out_path: str, name: str = "", packed_by: str = "Fizgig") -> dict:
    """Write a ReFLoRA: the LoRA at lora_path (a ReFLoRA's own references left out) plus `references`, a list of
    (RefMod file, tensor key) in order. Returns a summary for the caller's message."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    if not references:
        raise ValueError("a ReFLoRA needs at least one reference")
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"a ReFLoRA holds at most {MAX_REFERENCES} references")
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys() if not is_reference_key(k)}
    if not any(k.endswith((".lora_down.weight", ".lora_A.weight")) for k in tensors):
        raise ValueError(f"{os.path.basename(lora_path)} has no LoRA weights")
    metadata.pop(NODE_META_KEY, None)
    metadata.pop(HYBRID_KEY, None)
    n_lora = len(tensors)
    members, sources = [], []
    for i, (src, key) in enumerate(references):
        meta, tensor = _raw_member(src, key)
        for k in ("path", "bundle_index", "bundle_name", "tensor_key", "tokens", "bundle_audio"):
            meta.pop(k, None)
        meta["_format_version"] = NODE_FORMAT_VERSION
        members.append(meta)
        tensors[f"ref_{i}"] = tensor.contiguous()
        if os.path.basename(src) not in sources:
            sources.append(os.path.basename(src))
    display = name or os.path.splitext(os.path.basename(out_path))[0]
    metadata[NODE_META_KEY] = json.dumps({"_format_version": BUNDLE_FORMAT_VERSION, "kind": "bundle",
                                          "name": display, "members": members})
    metadata[HYBRID_KEY] = json.dumps({
        "version": HYBRID_VERSION,
        "lora": {"format": "kohya", "keys": n_lora, "network_dim": metadata.get("ss_network_dim"),
                 "network_alpha": metadata.get("ss_network_alpha")},
        "refmod_count": len(members),
        "packed_by": packed_by,
        "sources": {"lora": os.path.basename(lora_path), "refmods": sources},
    })
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    save_file(tensors, tmp, metadata=metadata)
    os.replace(tmp, out_path)
    return {"path": out_path, "lora_keys": n_lora, "references": len(members),
            "names": [str(m.get("name", f"ref_{i}")) for i, m in enumerate(members)]}
