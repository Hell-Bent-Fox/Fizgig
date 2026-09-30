"""Full fine-tune for described families - opt-in per driver.

A family trains its whole base model by rotation: the base stays frozen in NF4, and one component window at a time
(e.g. every block's attention, then every block's MLP gate ...) is swapped up to trainable bf16 from a CPU bf16
master, trained, and written back. A full rotation trains every component once. Checkpoints (and the previews that
ride them) happen only at whole rotations, so a saved file never has some components trained more than others.

A driver opts in by returning an FTSpec from `ft_spec(dit)`; a driver that doesn't has no fine-tune and never meets
this module. Everything here is model-agnostic: the window schedule and planner (shared with Krea 2's and H3's
original trainers), the NF4 swap per Linear (the format families/quant.py writes), the master read from the model
file, streaming of out-of-window blocks through the model's own block-swap hooks, and the streamed full-checkpoint
save in the source file's own layout.

Decided (Peter, 30 Sep 2026): component windows only, an NF4 trunk only, biases frozen, resume = continue from a
saved checkpoint (no optimizer state; the optimizer is rebuilt every window anyway).
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from fizgig.krea2.rotation import (RotationSchedule, component_entry_matches, component_gb_per_block,
                                   plan_component_windows, snap_ft_epochs)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FTSpec:
    """What a driver declares to offer a full fine-tune.

    blocks: the DiT attribute holding the numbered blocks (an nn.ModuleList).
    components: Linear-name prefixes within a block, in rotation order; each is one window spanning every block (the
        planner depth-splits a window that doesn't fit). Balance them by size - a window's bf16 weights, grads and
        optimizer state are what the card holds.
    always_on: dotted module names trained for the whole run (e.g. a text-fusion stack outside the blocks). Their
        Linears must be bf16 (outside the family's quant targets).
    overhead_gb: VRAM with the NF4 trunk resident plus activations and margin - the planner's base. None = the
        measured NF4 trunk + 3.5 GB.
    trunk_gb_per_block: NF4 trunk share per block (what streaming a block reclaims). None = measured.
    file_key: module weight name ("blocks.3.attn.wq.weight") -> the key in the model file, when the loader renames.
    """
    blocks: str = "blocks"
    components: tuple = ()
    always_on: tuple = ()
    overhead_gb: Optional[float] = None
    trunk_gb_per_block: Optional[float] = None
    slots_gb: float = 1.5
    file_key: Optional[Callable[[str], str]] = None


def source_unfit_reason(path: str) -> Optional[str]:
    """Why a model file cannot be fine-tuned (header read only), or None: a pre-quantised or fp8 file has no bf16
    layout to build the master from or write the checkpoint into."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
    with MemoryEfficientSafeOpen(path) as f:
        keys = f.keys()
        if any(k.endswith(".weight_scale") or k.endswith(".scale_weight") for k in keys):
            return "is a pre-quantized checkpoint (weight scale tensors)"
        low = [k for k in keys if str(f.header[k].get("dtype", "")).startswith(("F8", "I8", "U8"))]
    if low:
        return f"stores {len(low)} tensor(s) below 16 bits (e.g. {low[0]})"
    return None


def _unwrapped(name: str) -> str:
    """A Linear's name as the model file knows it: FamilyLoRA wraps each target and keeps the real Linear as `.base`."""
    return name[:-len(".base")] if name.endswith(".base") else name


def model_linears(module: nn.Module):
    """(name as the model file knows it, Linear) for the model's own Linears - never an adapter's (FamilyLoRA keeps
    its adapters under `.adapters.`, and their factors are Linears too)."""
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear) and ".adapters." not in f".{name}.":
            yield _unwrapped(name), m


class Rotator:
    """Swaps component windows between NF4-frozen and bf16-trainable, in place, on the real Linears.

    `master` (key -> CPU bf16) is the source of truth: a window activates FROM it and writes back TO it, so training
    never round-trips through NF4. FamilyLoRA's wrappers hold these Linears as `.base` and call them, so adapters
    (training adapter, context LoRA, speed LoRA) keep working while the weights underneath change.
    """

    def __init__(self, dit, spec: FTSpec, device):
        self.dit = dit
        self.spec = spec
        self.device = torch.device(device)
        self.blocks = dit.get_submodule(spec.blocks)
        self.key = spec.file_key or (lambda k: k)
        # every NF4 Linear in the blocks under a component prefix: (file key, linear, block index, name in block),
        # found once while everything is still frozen
        self.targets = []
        for bi, block in enumerate(self.blocks):
            for lname, m in model_linears(block):
                if not getattr(m, "_is_nf4", False):
                    continue
                if any(lname.startswith(p) for p in spec.components):
                    self.targets.append((self.key(f"{spec.blocks}.{bi}.{lname}.weight"), m, bi, lname))
        self.always = []                      # (file key, linear): dense bf16, trainable all run
        for mod_name in spec.always_on:
            mod = dit.get_submodule(mod_name)
            for lname, m in model_linears(mod):
                if getattr(m, "_is_nf4", False) or getattr(m, "_is_int8", False):
                    raise RuntimeError(f"[finetune] always-on module {mod_name}.{lname} is quantised - always-on "
                                       f"modules must stay bf16 (outside the family's quant targets)")
                self.always.append((self.key(f"{mod_name}.{lname}.weight"), m))
        self.master: Dict[str, torch.Tensor] = {}
        self.active: List = []
        self._forward = {}

    # ---- master ---------------------------------------------------------------------------------------------------
    def build_master(self, path: str) -> float:
        """Read every rotating weight from the model file (bf16 on disk), one tensor at a time. Never dequantised
        from the GPU copy: that has been through NF4, and the master is what gets trained and saved."""
        from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
        missing = []
        with MemoryEfficientSafeOpen(path) as src:
            have = set(src.keys())
            for key, _m, _bi, _ln in self.targets:
                if key not in have:
                    missing.append(key)
                    continue
                self.master[key] = src.get_tensor(key).to("cpu", dtype=torch.bfloat16).clone()
        if missing:
            raise RuntimeError(f"[finetune] {len(missing)} weights are not in {os.path.basename(path)} under the "
                               f"expected names, e.g. {missing[:3]} (the driver's FTSpec.file_key maps them)")
        gc.collect()
        return sum(v.numel() * v.element_size() for v in self.master.values()) / 1e9

    # ---- windows --------------------------------------------------------------------------------------------------
    def _window(self, spec) -> list:
        return [(k, m) for k, m, bi, ln in self.targets if any(component_entry_matches(c, ln, bi) for c in spec)]

    def resident_blocks(self, spec) -> set:
        """Blocks holding trainable Linears under a window (the streamer's resident set)."""
        n = len(self.blocks)
        out = set()
        for e in spec:
            out |= set(range(n)) if isinstance(e, str) else set(range(max(0, e[1]), min(n, e[2] + 1)))
        return out

    def _activate(self, pairs):
        for key, lin in pairs:
            self._forward[id(lin)] = lin.__dict__.pop("forward", None)    # the NF4 forward, restored verbatim
            lin.weight = nn.Parameter(self.master[key].to(self.device, dtype=torch.bfloat16), requires_grad=True)
            lin._nf4_packed = lin._nf4_state = None     # pure duplication while it trains
        return len(pairs)

    def _deactivate(self, pairs):
        from bitsandbytes.functional import quantize_nf4
        from fizgig.modules.nf4 import nf4_linear_forward_patch
        for key, lin in pairs:
            # the gradient first: the last step's autograd graph keeps this Parameter alive past the swap, and with
            # it a window-sized .grad (~6 GB on Krea 2's MLP windows) that freeing the weight storage does not free
            lin.weight.grad = None
            trained = lin.weight.detach()
            self.master[key] = trained.to("cpu", dtype=torch.bfloat16).clone()   # before the lossy re-encode
            packed, state = quantize_nf4(trained.contiguous())
            lin._nf4_packed, lin._nf4_state = packed, state
            lin.weight = nn.Parameter(torch.empty(0, device=packed.device, dtype=torch.bfloat16), requires_grad=False)
            saved = self._forward.pop(id(lin), None)
            lin.forward = saved if saved is not None else nf4_linear_forward_patch.__get__(lin, type(lin))
            # release the orphan: a C++-side autograd referrer keeps the old bf16 storage alive otherwise, and a
            # rotation would hold two windows (measured on H3 and Krea 2). Guarded: never free a storage still read.
            orphan = trained.untyped_storage()
            del trained
            if all(t is None or t.numel() == 0 or t.untyped_storage().data_ptr() != orphan.data_ptr()
                   for t in (lin.weight, lin._nf4_packed)):
                try:
                    orphan.resize_(0)
                except Exception:
                    pass
        return len(pairs)

    def rotate_to(self, spec) -> int:
        spec = list(spec)
        if spec == self.active:
            return 0
        if self.active:
            self._deactivate(self._window(self.active))
            gc.collect()
            torch.cuda.empty_cache()          # the outgoing window back to the allocator before the next one lands
        n = self._activate(self._window(spec)) if spec else 0
        self.active = spec
        if not spec:
            for _key, lin in self.always:     # parked: no gradient outlives the window it came from
                lin.weight.grad = None
        return n

    def start_always(self) -> int:
        for _key, lin in self.always:
            lin.weight.requires_grad_(True)   # weights only: biases stay frozen
        return len(self.always)

    def trainable_params(self) -> List[nn.Parameter]:
        return [lin.weight for _k, lin in self._window(self.active)] + [lin.weight for _k, lin in self.always]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Every trained weight in bf16 on the CPU: the master with the active window flushed in, plus always-on."""
        out = dict(self.master)
        for key, lin in self._window(self.active):
            out[key] = lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()
        for key, lin in self.always:
            out[key] = lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()
        return out


def _plan(comp_gb, n_blocks, trunk, spec: FTSpec, usable, allow_stream=True):
    overhead = spec.overhead_gb if spec.overhead_gb is not None else trunk * n_blocks + 3.5
    return plan_component_windows(usable, range(n_blocks), n_blocks, comp_gb, overhead_gb=overhead,
                                  trunk_gb_per_block=trunk, slots_gb=spec.slots_gb, allow_stream=allow_stream)


def plan_windows(dit, spec: FTSpec, rotator: Rotator, free_gb: float, allow_stream: bool = True):
    """(windows, stream, reasons, usable GB) for this card, with the model loaded: component sizes measured from it,
    `free_gb` read after the NF4 trunk landed (so the trunk is added back into the budget)."""
    block0 = dit.get_submodule(spec.blocks)[0]
    comp_gb = {p: 0.0 for p in spec.components}
    for ln, m in model_linears(block0):
        for p in spec.components:
            if ln.startswith(p):
                comp_gb[p] += m.out_features * m.in_features * 2 / 1e9     # logical size: NF4 empties .weight
                break
    n = len(dit.get_submodule(spec.blocks))
    if spec.trunk_gb_per_block is not None:
        trunk = float(spec.trunk_gb_per_block)
    else:
        packed = sum(lin._nf4_packed.numel() for _k, lin, _b, _l in rotator.targets
                     if getattr(lin, "_nf4_packed", None) is not None)
        trunk = packed / 1e9 / max(1, n)
    usable = free_gb + trunk * n - 1.5
    windows, stream, why = _plan(comp_gb, n, trunk, spec, usable, allow_stream)
    return windows, stream, why, usable


def plan_from_file(path: str, spec: FTSpec, free_gb: float):
    """The same plan before anything loads (the Training tab's "on this card" line): component sizes from the model
    file's header, `free_gb` read on the idle card. The trainer budgets after its model and preview VAE are in, so
    the non-block layers (kept bf16) and ~0.3 GB for the VAE come off here. Returns (windows, stream, usable) or
    None when the file does not show the spec's blocks."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
    import re
    with MemoryEfficientSafeOpen(path) as f:
        hdr = {k: f.header[k] for k in f.keys()}
    pat = re.compile(rf"^{re.escape(spec.blocks)}\.(\d+)\.(.+)\.weight$")
    blocks, comp_gb, block0_params, nonblock = set(), {p: 0.0 for p in spec.components}, 0, 0.0
    for k, info in hdr.items():
        shape = info.get("shape") or []
        numel = 1
        for d in shape:
            numel *= int(d)
        m = pat.match(k)
        if not m:
            nonblock += numel * 2 / 1e9
            continue
        blocks.add(int(m.group(1)))
        if m.group(1) != "0" or len(shape) != 2:
            continue
        block0_params += numel
        for p in spec.components:
            if m.group(2).startswith(p):
                comp_gb[p] += numel * 2 / 1e9
                break
    if not blocks or not all(comp_gb.values()):
        return None
    n = len(blocks)
    trunk = float(spec.trunk_gb_per_block) if spec.trunk_gb_per_block is not None else block0_params * 0.53 / 1e9
    usable = free_gb - nonblock - 0.3 - 1.5
    windows, stream, _why = _plan(comp_gb, n, trunk, spec, usable)
    return windows, stream, usable


def make_optimizer(params, lr):
    """Adafactor first (factored state ~10x smaller than Adam's - what keeps a full fine-tune on the card), then
    AdamW8bit, then AdamW. Returns (optimizer, label)."""
    try:
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False), "adafactor"
    except Exception:
        pass
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr), "adamw8bit"
    except Exception:
        return torch.optim.AdamW(params, lr=lr), "adamw"


class FusedSteps:
    """Optimizer in backward: one optimizer per parameter, stepped from its grad hook and the grad freed at once,
    so only one parameter's gradient is ever live (~5 GB saved on Krea 2). No clipping, no accumulation."""

    def __init__(self, lr):
        self.lr = lr
        self.opts = {}
        self.handles = []

    def attach(self, params):
        self.detach()
        for p in params:
            self.opts[p] = make_optimizer([p], self.lr)[0]

        def _hook(param):
            opt = self.opts.get(param)
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)
        for p in params:
            self.handles.append(p.register_post_accumulate_grad_hook(_hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.opts.clear()


def save_checkpoint(state: Dict[str, torch.Tensor], src_path: str, path: str, metadata: dict):
    """The fine-tuned model: the source file with the trained tensors replaced, in the source's own tensor order,
    streamed one tensor at a time (peak RAM ~ the master + one tensor, #143) to a .tmp renamed on completion (an
    interrupted save never leaves a truncated file under the real name)."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen, stream_save_file
    with MemoryEfficientSafeOpen(src_path) as src:
        header = {k: src.header[k] for k in src.keys()}
    extra = [k for k in state if k not in header]
    if extra:
        logger.warning("[finetune] %d trained tensor(s) have no slot in the source file and are NOT saved, e.g. %s",
                       len(extra), extra[:3])
    _dt = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
           "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
           "BOOL": torch.bool}
    reader = {"f": None}

    def producer(key):
        info = header[key]
        dt, shape = _dt[info["dtype"]], tuple(info["shape"])
        if key in state:
            return dt, shape, (lambda key=key, dt=dt, shape=shape: state[key].to(dt).reshape(shape))
        return dt, shape, (lambda key=key: reader["f"].get_tensor(key))

    specs = {k: producer(k) for k in header}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        reader["f"] = MemoryEfficientSafeOpen(src_path)
        try:
            stream_save_file(specs, tmp, metadata={str(k): str(v) for k, v in metadata.items()})
        finally:
            reader["f"].file.close()
        os.replace(tmp, path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    del specs
    gc.collect()
    return sum(1 for k in state if k in header), len(header)


def schedule(windows, n_blocks, rotate_every=1, start_window=0) -> RotationSchedule:
    return RotationSchedule(n_blocks, mode="component", components=tuple(windows), rotate_every=rotate_every,
                            start_window=start_window)


__all__ = ["FTSpec", "Rotator", "FusedSteps", "make_optimizer", "plan_windows", "save_checkpoint", "schedule",
           "snap_ft_epochs", "source_unfit_reason", "component_gb_per_block"]
