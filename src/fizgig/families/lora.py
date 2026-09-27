"""The standard layer's LoRA: wraps a family's Linears, trains one adapter and runs any number of frozen ones.

Which Linears (driver.block_map / lora_target_names) and how files are keyed (description.lora: file prefix, down/up
names, alpha key) come from the family, so every described family gets the same adapter machinery and writes its own
ComfyUI-compatible format.

Each wrapped Linear computes W x + sum_i s_i * B_i(A_i(x)). For a frozen adapter
    s = alpha / rank * load_strength * block_strength * (adapter on) * (block on)
where block_strength / block on belong to the block the module is in (driver.block_of); modules outside the block
map (e.g. a speed LoRA's modulation layers) follow the adapter's load strength and on/off only. The trainable adapter
is named "lora"; frozen ones (training adapter, context LoRA, a workbench primary/donor) get their own names.

Files in any common layout are accepted: the family's own keys, kohya (`lora_unet_<flattened>.lora_down/up`), or
PEFT / diffusers (`lora_A/lora_B` or `lora_down/lora_up`, bare or under `transformer.` / `diffusion_model.`).
LyCORIS (LoKR / LoHa) is not handled by the standard layer yet and is refused with a clear message.
"""
import math
import re

import torch
import torch.nn as nn

TRAINABLE = "lora"
_PREFIXES = ("transformer.", "diffusion_model.", "model.diffusion_model.", "base_model.model.", "")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.scales = {}

    def add(self, name, rank, alpha, trainable, A=None, B=None, strength=1.0):
        a = nn.Linear(self.base.in_features, rank, bias=False)
        b = nn.Linear(rank, self.base.out_features, bias=False)
        if A is not None:
            a.weight.data.copy_(A)
            b.weight.data.copy_(B)
        else:
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
        dev = self.base.weight.device
        dt = torch.float32 if trainable else torch.bfloat16
        a.to(dev, dt).requires_grad_(trainable)
        b.to(dev, dt).requires_grad_(trainable)
        self.adapters[name] = nn.Sequential(a, b)
        self.scales[name] = alpha / rank * strength

    def forward(self, x):
        out = self.base(x)
        for n, ad in self.adapters.items():
            s = self.scales.get(n, 0.0)
            if s:
                out = out + (s * ad(x.to(ad[0].weight.dtype))).to(out.dtype)
        return out


class FamilyLoRA:
    """A DiT's adapter set. `wrapped` maps the module name (relative to the DiT) -> LoRALinear."""

    def __init__(self, dit, driver):
        self.dit = dit
        self.driver = driver
        self.desc = driver.description
        self.targets = set(driver.lora_target_names(dit))
        self.linears = {n for n, m in dit.named_modules() if isinstance(m, nn.Linear)}
        self._flat = {n.replace(".", "_"): n for n in self.linears}
        self.wrapped = {}
        for full in sorted(self.targets):
            self._wrap(full)
        if not self.wrapped:
            raise RuntimeError(f"{self.desc.display_name}: none of the driver's LoRA targets exist in this model")
        # per frozen adapter: {"alpha_rank": {module: alpha/rank}, "load": float, "on": bool,
        #                      "block_mult": {block_id: float}, "block_on": {block_id: bool}}
        self._frozen = {}

    def _keys(self, full):
        """(down key, up key, alpha key) of any wrapped module, in the family's file format. Built from the module
        path alone, so modules outside the blocks never need a block id."""
        f = self.desc.lora
        stem = f"{f.file_prefix}{full}"
        return f"{stem}.{f.down}.weight", f"{stem}.{f.up}.weight", f.alpha_key.format(prefix=stem)

    def _wrap(self, full):
        """Wrap one Linear by dotted name (targets at init; frozen files may reach beyond them, e.g. a speed LoRA
        that also patches the modulation / timestep layers). Returns the LoRALinear or None."""
        if full in self.wrapped:
            return self.wrapped[full]
        parent_name, _, leaf = full.rpartition(".")
        parent = self.dit.get_submodule(parent_name) if parent_name else self.dit
        child = getattr(parent, leaf, None)
        if not isinstance(child, nn.Linear):
            return None
        w = LoRALinear(child)
        setattr(parent, leaf, w)
        self.wrapped[full] = w
        return w

    # ---- trainable ------------------------------------------------------------------------------
    def add_trainable(self, rank, alpha, blocks=None):
        """blocks: optional set of block ids to train (None = every target)."""
        for full, w in self.wrapped.items():
            if full not in self.targets:
                continue                    # extra Linears wrapped for a frozen file are never trained
            if blocks is None or self.driver.block_of(full) in blocks:
                w.add(TRAINABLE, rank, alpha, True)
        self.rank, self.alpha = rank, alpha

    def trainable_modules(self):
        return nn.ModuleList([w.adapters[TRAINABLE] for w in self.wrapped.values() if TRAINABLE in w.adapters])

    def parameters(self):
        return [p for m in self.trainable_modules() for p in m.parameters()]

    # ---- reading LoRA files in any common layout ------------------------------------------------
    def _module_for(self, stem):
        """A file's module stem (dotted, prefixed, or kohya-flattened) -> a Linear name in this model, or None."""
        if stem.startswith("lora_unet_"):
            return self._flat.get(stem[len("lora_unet_"):])
        for p in _PREFIXES:
            if p and not stem.startswith(p):
                continue
            name = stem[len(p):]
            if name in self.linears:
                return name
            if name.replace(".", "_") in self._flat:
                return self._flat[name.replace(".", "_")]
        return None

    def read_file(self, path):
        """-> {module name: (A, B, alpha)} for every Linear the file adapts in this model."""
        from safetensors.torch import load_file
        sd = load_file(path)
        if any(re.search(r"\.(lokr_w1|lokr_w2|hada_w1_a)(\.|$)", k) for k in sd):
            raise ValueError(f"{path}: LoKR / LoHa files are not supported by the standard layer yet")
        out = {}
        for key in sd:
            m = re.match(r"(.+)\.(lora_A|lora_down)\.weight$", key)
            if not m:
                continue
            stem, down = m.group(1), m.group(2)
            up = "lora_B" if down == "lora_A" else "lora_up"
            if f"{stem}.{up}.weight" not in sd:
                continue
            full = self._module_for(stem)
            if full is None:
                continue
            A, B = sd[key], sd[f"{stem}.{up}.weight"]
            alpha = sd.get(f"{stem}.alpha")
            out[full] = (A, B, float(alpha.item()) if alpha is not None else float(A.shape[0]))
        return out

    # ---- frozen adapters ------------------------------------------------------------------------
    def add_file(self, path, name, strength=1.0):
        """Attach a LoRA file frozen under `name` on every Linear it adapts. Returns the number of Linears covered
        (0 = nothing in the file matches this model)."""
        n = 0
        ar = {}
        for full, (A, B, alpha) in self.read_file(path).items():
            w = self._wrap(full)
            if w is None or A.shape[1] != w.base.in_features or B.shape[0] != w.base.out_features:
                continue
            w.add(name, A.shape[0], alpha, False, A, B)
            ar[full] = alpha / A.shape[0]
            n += 1
        self._frozen[name] = {"alpha_rank": ar, "load": float(strength), "on": True, "block_mult": {},
                              "block_on": {}, "path": path}
        self._apply(name)
        return n

    def has(self, name):
        return name in self._frozen

    def _apply(self, name):
        st = self._frozen[name]
        for full, ar in st["alpha_rank"].items():
            b = self.driver.block_of(full)
            on = st["on"] and (b is None or st["block_on"].get(b, True))
            mult = 1.0 if b is None else st["block_mult"].get(b, 1.0)
            self.wrapped[full].scales[name] = ar * st["load"] * mult if on else 0.0

    def set_enabled(self, name, enabled: bool):
        """Switch a frozen adapter on (with its load strength and block settings) or fully off - every module,
        inside or outside the block map."""
        if name in self._frozen:
            self._frozen[name]["on"] = bool(enabled)
            self._apply(name)

    def set_strength(self, name, strength):
        """The adapter's whole-file (load) strength."""
        self._frozen[name]["load"] = float(strength)
        self._apply(name)

    def set_blocks(self, name, mult=None, enabled=None):
        """Per-block controls for one adapter: mult {block_id: strength}, enabled {block_id: bool}. Blocks not
        named keep their current values."""
        st = self._frozen[name]
        st["block_mult"].update(mult or {})
        st["block_on"].update(enabled or {})
        self._apply(name)

    def adapter_blocks(self, name):
        """Block ids the adapter touches (the workbench greys out the rest)."""
        return {b for b in (self.driver.block_of(f) for f in self._frozen.get(name, {}).get("alpha_rank", {}))
                if b is not None}

    @torch.no_grad()
    def swap_file(self, name, path):
        """Replace an adapter's weights with another file's (epoch scrubbing). In place when the file adapts the
        same modules at the same ranks; otherwise the adapter is rebuilt. Block settings and strength carry over.
        Returns the number of Linears covered."""
        st = self._frozen[name]
        new = self.read_file(path)
        same = set(new) == set(st["alpha_rank"]) and all(
            self.wrapped[f].adapters[name][0].weight.shape == new[f][0].shape for f in new)
        if same:
            for full, (A, B, alpha) in new.items():
                a, b = self.wrapped[full].adapters[name]
                a.weight.copy_(A.to(a.weight.dtype))
                b.weight.copy_(B.to(b.weight.dtype))
                st["alpha_rank"][full] = alpha / A.shape[0]
            st["path"] = path
            self._apply(name)
            return len(new)
        keep = {k: st[k] for k in ("load", "on", "block_mult", "block_on")}
        self.remove(name)
        n = self.add_file(path, name, keep["load"])
        self._frozen[name].update(keep)
        self._apply(name)
        return n

    def remove(self, name):
        """Drop a frozen adapter's weights entirely."""
        for w in self.wrapped.values():
            if name in w.adapters:
                del w.adapters[name]
                w.scales.pop(name, None)
        self._frozen.pop(name, None)

    def move_adapter(self, name, device):
        """Move one frozen adapter's weights (e.g. a speed LoRA parked on CPU between previews)."""
        for w in self.wrapped.values():
            if name in w.adapters:
                w.adapters[name].to(device)

    # ---- bake: what is live now, as one standard LoRA in the family format ------------------------
    @torch.no_grad()
    def bake(self, names, dtype=torch.bfloat16):
        """One LoRA state dict equal to the named adapters as currently set (strengths, block settings, on/off),
        in the family's key format. Several adapters on a module are rank-concatenated with their scales folded
        into the up weights, so alpha = total rank (scale 1). Returns (state_dict, {module: rank})."""
        sd, ranks = {}, {}
        for full, w in self.wrapped.items():
            As, Bs = [], []
            for n in names:
                s = w.scales.get(n, 0.0) if n in w.adapters else 0.0
                if not s:
                    continue
                a, b = w.adapters[n]
                As.append(a.weight.float())
                Bs.append(b.weight.float() * s)
            if not As:
                continue
            A, B = torch.cat(As, 0), torch.cat(Bs, 1)
            ka, kb, kal = self._keys(full)
            sd[ka] = A.to("cpu", dtype).contiguous()
            sd[kb] = B.to("cpu", dtype).contiguous()
            sd[kal] = torch.tensor(float(A.shape[0]))
            ranks[full] = A.shape[0]
        return sd, ranks

    # ---- save / load the trainable adapter (approved key format) --------------------------------
    def state_dict(self, dtype=torch.bfloat16):
        sd = {}
        for full, w in self.wrapped.items():
            if TRAINABLE in w.adapters:
                a, b = w.adapters[TRAINABLE]
                ka, kb, kal = self._keys(full)
                sd[ka] = a.weight.detach().to("cpu", dtype).contiguous()
                sd[kb] = b.weight.detach().to("cpu", dtype).contiguous()
                sd[kal] = torch.tensor(float(self.alpha))
        return sd

    def save(self, path, metadata=None, dtype=torch.bfloat16):
        from safetensors.torch import save_file
        save_file(self.state_dict(dtype), path, metadata={k: str(v) for k, v in (metadata or {}).items()})

    @torch.no_grad()
    def load_trainable(self, path):
        """Restore the trainable adapter from a file saved by save(). Returns modules matched."""
        from safetensors.torch import load_file
        sd = load_file(path)
        n = 0
        for full, w in self.wrapped.items():
            if TRAINABLE not in w.adapters:
                continue                # frozen-only wraps (training adapter, speed LoRA extras) hold nothing to load
            ka, kb, _ = self._keys(full)
            if ka in sd:
                a, b = w.adapters[TRAINABLE]
                a.weight.copy_(sd[ka].to(a.weight.dtype))
                b.weight.copy_(sd[kb].to(b.weight.dtype))
                n += 1
        return n
