"""The standard layer's LoRA: wraps a family's target Linears, trains one adapter and runs frozen ones alongside.

Which Linears (driver.lora_target_names) and how keys are written (description.lora: key template, down/up
names, alpha key) come from the family - so every described family gets the same adapter machinery and its own
ComfyUI-compatible file format.

Each wrapped Linear computes W x + sum_i s_i * B_i(A_i(x)), s_i = alpha_i / rank_i * strength_i.
The trainable adapter is named "lora"; frozen ones (the training adapter, a context LoRA) get their own
names and can be switched off (scale 0) without being unloaded, which is how previews run adapter-off.
"""
import math

import torch
import torch.nn as nn

TRAINABLE = "lora"


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
        targets = set(driver.lora_target_names(dit))
        self.targets = targets
        self.wrapped = {}
        for name, mod in list(dit.named_modules()):
            for cname, child in list(mod.named_children()):
                full = f"{name}.{cname}" if name else cname
                if isinstance(child, nn.Linear) and full in targets:
                    w = LoRALinear(child)
                    setattr(mod, cname, w)
                    self.wrapped[full] = w
        if not self.wrapped:
            raise RuntimeError(f"{self.desc.display_name}: none of the driver's LoRA targets exist in this model")
        self._base_scale = {}

    def _keys(self, full):
        """(down key, up key, alpha key) for a wrapped module, in the family's file format."""
        f = self.desc.lora
        block = self.driver.block_of(full)
        module = full[len(self.desc.block_prefix) + 1:].split(".", 1)[1]
        prefix = self.desc.lora_prefix(block, module)
        return f.key(block, module, "down"), f.key(block, module, "up"), f.alpha_key.format(prefix=prefix)

    # ---- trainable ------------------------------------------------------------------------------
    def add_trainable(self, rank, alpha, blocks=None):
        """blocks: optional set of block indices to train (None = all)."""
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

    # ---- frozen file adapters -------------------------------------------------------------------
    def _wrap(self, full):
        """Wrap one more Linear by dotted name (for frozen files that reach beyond the LoRA targets, e.g. a speed
        LoRA that also patches the modulation / timestep layers). Returns the LoRALinear or None."""
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

    def add_file(self, path, name, strength=1.0):
        """Attach a LoRA file frozen, on every Linear the file names (in the family's key format). Returns the
        number of Linears covered (0 = wrong format)."""
        from safetensors.torch import load_file
        f = self.desc.lora
        sd = load_file(path)
        n = 0
        for key in list(sd):
            full = f.module_of(key)
            if full is None:
                continue
            w = self._wrap(full)
            if w is None:
                continue
            A, B = sd[key], sd[f"{f.file_prefix}{full}.{f.up}.weight"]
            r = A.shape[0]
            ak = f.alpha_key.format(prefix=f"{f.file_prefix}{full}")
            alpha = float(sd[ak].item()) if ak in sd else r
            w.add(name, r, alpha, False, A, B, strength)
            n += 1
        self._base_scale[name] = {full: w.scales[name] for full, w in self.wrapped.items() if name in w.adapters}
        return n

    def move_adapter(self, name, device):
        """Move one frozen adapter's weights (e.g. a speed LoRA parked on CPU between previews)."""
        for w in self.wrapped.values():
            if name in w.adapters:
                w.adapters[name].to(device)

    def set_enabled(self, name, enabled: bool):
        """Frozen adapters only: switch a named adapter on (its loaded strength) or off."""
        for full, s in self._base_scale.get(name, {}).items():
            self.wrapped[full].scales[name] = s if enabled else 0.0

    # ---- save / load (approved key format) ------------------------------------------------------
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
            ka, kb, _ = self._keys(full)
            if TRAINABLE in w.adapters and ka in sd:
                a, b = w.adapters[TRAINABLE]
                a.weight.copy_(sd[ka].to(a.weight.dtype))
                b.weight.copy_(sd[kb].to(b.weight.dtype))
                n += 1
        return n
