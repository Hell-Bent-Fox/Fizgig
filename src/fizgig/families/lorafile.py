"""Reading a LoRA file for a described family without loading the model (Profiler, Extract).

Resolves every LoRA pair in the file - the family's own keys, kohya (`lora_unet_<flattened>`), or PEFT / diffusers
(`lora_A/B` or `lora_down/up`, bare or under a common prefix) - to the dotted module name the family uses. Dotted names
resolve for any module; kohya's flattened names can only be un-flattened for modules the block map knows.
"""
import re

_DOWN = re.compile(r"(.+)\.(lora_A|lora_down)\.weight$")
_PREFIXES = ("transformer.", "diffusion_model.", "model.diffusion_model.", "base_model.model.")
_LOHA = re.compile(r"\.hada_w1_a(\.|$)")
_LOKR = re.compile(r"(.+)\.lokr_w1(_a)?$")


def lora_pairs(desc, keys):
    """-> [(module or None, down key, up key, alpha key or None)] for every down/up pair in `keys`. module is None
    for a kohya-flattened name outside the block map (it cannot be un-flattened without the model)."""
    keys = set(keys)
    if any(_LOHA.search(k) for k in keys):
        raise ValueError("LoHa files are not supported by the standard layer yet")
    drv = desc.load_driver()
    known = {m for g in drv.block_map() for b in g.blocks for m in b.modules}
    flat = {m.replace(".", "_"): m for m in known}
    out = []
    for key in sorted(keys):
        m = _DOWN.match(key)
        if not m:
            continue
        stem = m.group(1)
        up = f"{stem}.{'lora_B' if m.group(2) == 'lora_A' else 'lora_up'}.weight"
        if up not in keys:
            continue
        if stem.startswith("lora_unet_"):
            mod = flat.get(stem[len("lora_unet_"):])
        else:
            mod = next((stem[len(p):] for p in _PREFIXES if stem.startswith(p)), stem)
        alpha = f"{stem}.alpha"
        out.append((mod, key, up, alpha if alpha in keys else None))
    return out


def lokr_modules(desc, keys):
    """-> [(module or None, stem)] for every LoKR module in `keys` (stem = the key prefix before .lokr_*)."""
    keys = set(keys)
    known = {m for g in desc.load_driver().block_map() for b in g.blocks for m in b.modules}
    flat = {m.replace(".", "_"): m for m in known}
    out = []
    for key in sorted(keys):
        m = _LOKR.match(key)
        if not m:
            continue
        stem = m.group(1)
        if stem.startswith("lora_unet_"):
            mod = flat.get(stem[len("lora_unet_"):])
        else:
            mod = next((stem[len(p):] for p in _PREFIXES if stem.startswith(p)), stem)
        out.append((mod, stem))
    return out


def lokr_factors(f, stem):
    """(w1, w2, scale) of a LoKR module from an open safetensors file, low-rank factors multiplied out."""
    from fizgig.networks.lora import lycoris_scale_from_keys
    keys = {k[len(stem) + 1:]: f.get_tensor(k) for k in f.keys() if k.startswith(stem + ".")}
    w1 = keys["lokr_w1"] if "lokr_w1" in keys else keys["lokr_w1_a"].float() @ keys["lokr_w1_b"].float()
    w2 = keys["lokr_w2"] if "lokr_w2" in keys else keys["lokr_w2_a"].float() @ keys["lokr_w2_b"].float()
    return w1.float(), w2.float(), lycoris_scale_from_keys(keys)


def block_of(desc):
    """{dotted module: block id} for the family's block map."""
    return {m: b.id for g in desc.load_driver().block_map() for b in g.blocks for m in b.modules}


def family_keys(desc, module):
    """(down, up, alpha) keys of a module in the family's own file format."""
    f = desc.lora
    stem = f"{f.file_prefix}{module}"
    return f"{stem}.{f.down}.weight", f"{stem}.{f.up}.weight", f.alpha_key.format(prefix=stem)
