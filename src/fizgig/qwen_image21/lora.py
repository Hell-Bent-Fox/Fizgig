"""Qwen Image 2.1 LoRA: wraps the block Linears, trains one adapter and runs frozen ones alongside.

Keys follow the family's approved exception to the kohya rule (families/qwen_image.py LoRAFormat):
`transformer.transformer_blocks.{N}.{module}.lora_A/lora_B.weight` + `.alpha`. ComfyUI maps
gate_layer/proj onto the halves of its fused gate_up only for bare or `transformer.`-prefixed keys.

Each wrapped Linear computes W x + sum_i s_i * B_i(A_i(x)), s_i = alpha_i / rank_i * strength_i.
The trainable adapter is named "lora"; frozen ones (the training adapter, a context LoRA) get their own
names and can be switched off (scale 0) without being unloaded, which is how previews run adapter-off.
"""
import math
import re

import torch
import torch.nn as nn

TARGET_RE = re.compile(r"transformer_blocks\.(\d+)\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|"
                       r"img_mlp\.gate_layer|img_mlp\.proj|img_mlp\.out)")
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


class QwenLoRA:
    """The wrapped DiT's adapter set. `wrapped` maps 'transformer_blocks.N.module' -> LoRALinear."""

    def __init__(self, dit):
        self.wrapped = {}
        for name, mod in list(dit.named_modules()):
            for cname, child in list(mod.named_children()):
                full = f"{name}.{cname}" if name else cname
                if isinstance(child, nn.Linear) and TARGET_RE.fullmatch(full):
                    w = LoRALinear(child)
                    setattr(mod, cname, w)
                    self.wrapped[full] = w
        self._base_scale = {}

    # ---- trainable ------------------------------------------------------------------------------
    def add_trainable(self, rank, alpha, blocks=None):
        """blocks: optional set of block indices to train (None = all)."""
        for full, w in self.wrapped.items():
            if blocks is None or int(full.split(".")[1]) in blocks:
                w.add(TRAINABLE, rank, alpha, True)
        self.rank, self.alpha = rank, alpha

    def trainable_modules(self):
        return nn.ModuleList([w.adapters[TRAINABLE] for w in self.wrapped.values() if TRAINABLE in w.adapters])

    def parameters(self):
        return [p for m in self.trainable_modules() for p in m.parameters()]

    # ---- frozen file adapters -------------------------------------------------------------------
    def add_file(self, path, name, strength=1.0):
        """Attach a LoRA file frozen. Returns the number of Linears it covered (0 = wrong format)."""
        from safetensors.torch import load_file
        sd = load_file(path)
        n = 0
        for full, w in self.wrapped.items():
            ka = f"transformer.{full}.lora_A.weight"
            if ka not in sd:
                continue
            A, B = sd[ka], sd[f"transformer.{full}.lora_B.weight"]
            r = A.shape[0]
            alpha = float(sd[f"transformer.{full}.alpha"].item()) if f"transformer.{full}.alpha" in sd else r
            w.add(name, r, alpha, False, A, B, strength)
            n += 1
        self._base_scale[name] = {full: w.scales[name] for full, w in self.wrapped.items() if name in w.adapters}
        return n

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
                sd[f"transformer.{full}.lora_A.weight"] = a.weight.detach().to("cpu", dtype).contiguous()
                sd[f"transformer.{full}.lora_B.weight"] = b.weight.detach().to("cpu", dtype).contiguous()
                sd[f"transformer.{full}.alpha"] = torch.tensor(float(self.alpha))
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
            ka = f"transformer.{full}.lora_A.weight"
            if TRAINABLE in w.adapters and ka in sd:
                a, b = w.adapters[TRAINABLE]
                a.weight.copy_(sd[ka].to(a.weight.dtype))
                b.weight.copy_(sd[f"transformer.{full}.lora_B.weight"].to(b.weight.dtype))
                n += 1
        return n
