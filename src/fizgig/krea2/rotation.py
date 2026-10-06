"""Rotating-block fine-tuning for Krea 2 — full-rank updates on a consumer card.

Naive full fine-tuning of Krea 2 needs ~78 GB (bf16 weights + grads + Adam state).
This trains a *rotating slice* of the model instead: the DiT stays fp8-frozen
(~14 GB resident) and only N blocks at a time are swapped up to trainable bf16.
Over a cycle every block gets trained, but VRAM only ever holds grads and
optimizer state for the active slice.

The technique is LISA (Layerwise Importance Sampled AdamW) applied to a diffusion
DiT rather than an LLM — unproven in this setting, hence the experiment branch.

Precision: a CPU-resident bf16 **master copy** is the source of truth for every
trainable weight. Blocks activate FROM the master and write back TO it, so the
lossy fp8 copy on the GPU is only ever used for the frozen forward pass — training
never round-trips through fp8, which would quantize away exactly the small updates
we're trying to learn.
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RotationSchedule:
    """Which block indices are trainable at a given epoch.

    `active` blocks train for `rotate_every` epochs, then the window advances.
    A full cycle (every block trained once) takes ceil(n_blocks / active) *
    rotate_every epochs — worth knowing when picking max_train_epochs, since
    fewer than one full cycle means part of the model never trains at all.
    """

    def __init__(self, n_blocks: int, active: int = 4, rotate_every: int = 1,
                 order: Optional[Sequence[int]] = None, mode: str = "block",
                 components: Sequence[str] = ("attn", "mlp.gate", "mlp.up", "mlp.down"),
                 start_window: int = 0):
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        self.n_blocks = int(n_blocks)
        self.active = max(1, min(int(active), self.n_blocks))
        self.rotate_every = max(1, int(rotate_every))
        # A resumed FT run is a fresh process: without an offset it restarts the cycle at
        # window 0, retraining early windows and starving late ones. start_window shifts the
        # whole schedule so "resume for the last epoch" can mean "train the last component".
        self.start_window = int(start_window)
        self.order = list(order) if order is not None else list(range(self.n_blocks))
        if sorted(self.order) != list(range(self.n_blocks)):
            raise ValueError("order must be a permutation of range(n_blocks)")
        # Default components are balanced by parameter count: a block is 30% attention and
        # 70% MLP, so a naive ("attn", "mlp") split makes the MLP window 2.3x the attention
        # one — big enough to OOM. Splitting the MLP into its three matrices gives four
        # windows of <=30% each.
        # "block": windows are contiguous slices of blocks (depth-split).
        # "component": windows are sub-block components spanning EVERY block — attention
        # across the whole model, then MLP. Roughly the same VRAM (attention is about half
        # a block) but each window sees the model's full depth, so a concept is learned by
        # every layer at once instead of by one depth slice at a time.
        self.mode = "component" if str(mode).startswith("comp") else "block"
        self.components = list(components)

    @property
    def n_windows(self) -> int:
        if self.mode == "component":
            return len(self.components)
        return (self.n_blocks + self.active - 1) // self.active

    @property
    def cycle_epochs(self) -> int:
        """Epochs needed for every block to have trained once."""
        return self.n_windows * self.rotate_every

    def window_at(self, epoch: int) -> int:
        """0-based window index for a 0-based epoch (wraps after a full cycle)."""
        return ((int(epoch) // self.rotate_every) + self.start_window) % self.n_windows

    def active_at(self, epoch: int):
        """Block indices (block mode) or component window entries (component mode) — an
        entry is a bare prefix string (full depth) or a (prefix, lo, hi) depth-split."""
        w = self.window_at(epoch)
        if self.mode == "component":
            c = self.components[w]
            return list(c) if isinstance(c, list) else [c]     # a list: several parts trained together
        start = w * self.active
        return sorted(self.order[start:start + self.active])

    def describe(self) -> str:
        if self.mode == "component":
            _names = ["+".join(c) if isinstance(c, list) else c if isinstance(c, str) else f"{c[0]}@{c[1]}-{c[2]}"
                      for c in self.components]
            return (f"component windows {_names} across all {self.n_blocks} blocks, "
                    f"rotating every {self.rotate_every} epoch(s) — {self.n_windows} windows, "
                    f"{self.cycle_epochs} epochs per full cycle")
        return (f"{self.active} of {self.n_blocks} blocks per window, rotating every "
                f"{self.rotate_every} epoch(s) — {self.n_windows} windows, "
                f"{self.cycle_epochs} epochs per full cycle")


def component_gb_per_block(block: nn.Module, prefixes) -> dict:
    """Exact bf16 GB per block for each component prefix, measured from the model itself.

    Sums Linear weight numel x 2 bytes under each prefix — no architecture guessing, so
    the window planner's sizes are exact for whatever checkpoint actually loaded."""
    out = {p: 0.0 for p in prefixes}
    for name, m in block.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        for p in prefixes:
            if name.startswith(p):
                # Logical shape, NOT weight.numel(): what we are sizing is the TRAINABLE
                # bf16 window, which is the same however the frozen copy is stored — and
                # NF4 empties `.weight` to a 0-element tensor, so measuring it reported
                # every component as 0 GB and the planner divided by zero.
                out[p] += m.out_features * m.in_features * 2 / 1e9
                break
    return out


def snap_ft_epochs(max_epochs, cycle_epochs, start_window=0, rotate_every=1):
    """Snap a fine-tune's epoch count UP so the run ENDS on a rotation-cycle boundary.

    An off-cycle total leaves the final checkpoint with some components trained one more
    pass than others - and the final save is the one most people keep. Snap UP, never
    down: the typed number expressed how much training the user wants at minimum (same
    doctrine as retirement stops and the save cadence).

    Start-aware for pause/resume: a resumed leg begins mid-cycle at start_window, so the
    snap aligns start + length to a boundary - which means a leg resumed from a run whose
    ORIGINAL total was cycle-aligned snaps by zero, and pause/resume keeps landing on the
    original total. Pure, so the arithmetic is pinnable without a card."""
    m = int(max_epochs)
    cyc = max(1, int(cycle_epochs))
    if m <= 0:
        return m
    pos = (int(start_window or 0) * max(1, int(rotate_every))) % cyc
    rem = (pos + m) % cyc
    return m if rem == 0 else m + (cyc - rem)

def plan_component_windows(usable_gb, span, n_blocks, comp_gb, overhead_gb,
                           trunk_gb_per_block, slots_gb, allow_stream=True,
                           max_sane_windows=12, spans=None, trunk_credit=True, pack_margin_gb=0.0,
                           max_parts=0):
    """The family-agnostic component-window plan: (windows, stream, reasons).

    comp_gb: ordered (prefix -> bf16 GB per block). windows: RotationSchedule component
    entries — bare prefixes where the full span fits, (prefix, lo, hi) depth-splits where
    it doesn't, and lists of prefixes where several whole parts fit one window (the fewest
    windows the card holds; one list of every part = the whole model at once). stream: True when the frozen out-of-window blocks must stream from CPU to
    fit. Returns (None, stream, reasons) when the budget can't run FT at all.

    Depth-splitting trades the full-depth-per-window geometry (what makes component mode
    learn fast) for fit; streaming further trades step speed (PCIe) for residency. Both
    families feed their own calibrated constants; the trainer's per-window peak logs are
    what refine them. Pure so the tier tables are pinnable without a card.

    spans: prefix -> the blocks that hold it, where a component lives in some blocks only (Klein: img_attn in the
    double blocks, linear1 in the single ones); comp_gb is then per block OF ITS SPAN, and a depth-split divides
    that span, never into blocks without it. trunk_credit / pack_margin_gb: how a window of several parts is sized
    (H3's backend measured over the credited plan, so it packs without the credit and with a margin). max_parts:
    the user's cap on parts per window (0 = as many as fit; 1 = one part per window, the most headroom)."""
    span = sorted(int(b) for b in span)
    spans = {p: sorted(int(b) for b in s) for p, s in (spans or {}).items()}

    def span_of(prefix):
        return spans.get(prefix, span)
    usable = float(usable_gb)
    prefixes = list(comp_gb)
    fattest = max(comp_gb.values())
    reasons = []

    def _chunk(prefix, max_gb):
        """Even depth-chunks of `span` whose bf16 window fits max_gb, or None."""
        g = comp_gb[prefix]
        if g <= 0:
            # A component that measures as weightless means the sizer could not see this
            # storage scheme — fail loudly rather than divide by zero or, worse, plan a
            # cycle for windows it thinks are free.
            raise ValueError(f"component {prefix!r} measured 0 GB per block — the window "
                             f"sizer does not understand this model's weight storage")
        max_len = int(max_gb / g)
        if max_len < 1:
            return None
        sp = span_of(prefix)
        k = (len(sp) + max_len - 1) // max_len
        per = (len(sp) + k - 1) // k
        chunks = [sp[i:i + per] for i in range(0, len(sp), per)]
        if k == 1:
            return [prefix]                       # full span — bare prefix, full depth
        return [(prefix, c[0], c[-1]) for c in chunks]

    # Packed plan: as many whole parts per window as fit, the fewest windows first. A trained Linear drops its
    # quantised copy, so a window of several parts is credited their share of the trunk (measured 5 Oct at 1 MP:
    # Anima and Qwen 2.1 trained whole within 0.2 / 0.7 GB under this); a one-part window keeps the calibrated
    # overhead + its weights.
    full = {p: len(span_of(p)) * comp_gb[p] for p in prefixes}
    n_span = len(set().union(*spans.values())) if spans else len(span)
    per_gb = trunk_gb_per_block * n_span / max(1e-9, sum(full.values()))    # trunk GB per bf16 GB trained

    def need(group):
        w = sum(full[p] for p in group)
        if len(group) == 1:
            return overhead_gb + w
        return overhead_gb + w - (per_gb * w if trunk_credit else 0.0) + pack_margin_gb

    groups = []
    for p in sorted(prefixes, key=lambda q: -full[q]):        # first-fit decreasing
        for g in groups:
            if (not max_parts or len(g) < max_parts) and need(g + [p]) <= usable:
                g.append(p)
                break
        else:
            groups.append([p])
    if len(groups) < len(prefixes) and all(need(g) <= usable for g in groups):
        order = {p: i for i, p in enumerate(prefixes)}
        groups = sorted((sorted(g, key=order.get) for g in groups), key=lambda g: order[g[0]])
        reasons.append(f"{len(prefixes)} parts packed into {len(groups)} window(s), each at full depth (largest "
                       f"~{max(need(g) for g in groups):.1f} GB of {usable:.1f} usable)")
        return [g if len(g) > 1 else g[0] for g in groups], False, reasons

    # Full-speed plan: everything resident, split only what the budget forces. Viable
    # only when at least a 1-block window of the fattest component fits the cap.
    cap = usable - overhead_gb
    plain = []
    if cap >= fattest:
        for prefix in prefixes:
            _c = _chunk(prefix, cap)
            if _c is None:
                plain = []
                break
            plain.extend(_c)
    if plain and len(plain) <= max_sane_windows:
        for prefix in prefixes:
            need = overhead_gb + len(span_of(prefix)) * comp_gb[prefix]
            if need > usable:
                reasons.append(f"{prefix} across {len(span_of(prefix))} block(s) would peak "
                               f"~{need:.1f} GB vs {usable:.1f} usable — depth-split")
        return plain, False, reasons

    if not allow_stream:
        reasons.append(f"{usable:.1f} GB usable cannot hold the frozen trunk + a component "
                       f"window at full residency, and streaming is disabled")
        return (plain or None), False, reasons

    # Streaming plan: only the active chunk's blocks stay resident; every other block's
    # frozen weights ring in from CPU. Per-chunk peak ≈ overhead − reclaimed + slots + window.
    base = overhead_gb - trunk_gb_per_block * int(n_blocks) + slots_gb
    stream_windows = []
    for prefix in prefixes:
        g = comp_gb[prefix]
        per_block = g + trunk_gb_per_block         # window bf16 + the block's own frozen share
        max_gb = (usable - base) / per_block * g   # expressed back in window GB
        chunks = _chunk(prefix, max_gb) if max_gb > 0 else None
        if chunks is None:
            reasons.append(f"{usable:.1f} GB usable cannot fit even a 1-block {prefix} "
                           f"window with streaming (~{base + per_block:.1f} GB needed)")
            return None, True, reasons
        stream_windows.extend(chunks)
    reasons.append(f"{usable:.1f} GB usable is below the resident plan — frozen "
                   f"out-of-window blocks stream from CPU "
                   f"(~{trunk_gb_per_block * n_blocks:.1f} GB staged in RAM, "
                   f"steps slower for the PCIe trips)")
    return stream_windows, True, reasons


# Krea 2 component-FT VRAM model. One measured anchor exists: 27.67 GB peak on the 5090
# for classic full-depth component windows — but that peak IS the rotation boundary,
# where the pre-fix rotation transiently held BOTH windows (~7 GB over steady state).
# With the boundary ref-drop fix (ported from H3) the steady model is fp8 trunk ~13 GB
# + activations + margin. Conservative until the trainer's per-window peak logs
# calibrate: OVERHEAD 16.0 (trunk 13 + act ~1.5 + margin), fp8 reclaimed per streamed
# block 13/28, JIT-streamer in-flight allowance 1.5 (RotationOffloader prefetches one
# block, no flat ring slots).
K2FT_COMPONENT_PREFIXES = ("attn", "mlp.gate", "mlp.up", "mlp.down")
_K2FT_OVERHEAD_GB = 16.0        # fp8 trunk (~13 GB) + activations + margin
_K2FT_FP8_GB_PER_BLOCK = 0.464  # 13.0 / 28
# NF4 trunk: measured 6.08 GB packed for the same 224 Linears (24.31 GB of bf16/fp8
# freed), i.e. the frozen base costs less than half. The overhead constant drops by the
# same difference; both are still estimates until the per-window peak lines confirm them.
_K2FT_NF4_OVERHEAD_GB = 9.5
_K2FT_NF4_GB_PER_BLOCK = 0.217  # 6.08 / 28
_K2FT_STREAM_SLOTS_GB = 1.5


def krea2_trunk_gb_per_block(nf4: bool = False) -> float:
    """Frozen-trunk cost per block, for the caller that has to add it back to `usable`."""
    return _K2FT_NF4_GB_PER_BLOCK if nf4 else _K2FT_FP8_GB_PER_BLOCK


def plan_krea2_ft_windows(usable_gb, comp_gb, n_blocks=28, allow_stream=True,
                          nf4: bool = False):
    """The Krea 2 component-window plan: (windows, stream, reasons).

    comp_gb comes from component_gb_per_block on a real block (exact, no architecture
    guessing); the constants above are the calibration. `nf4` selects the 4-bit trunk's
    much smaller footprint, which is what lets a small card keep its window RESIDENT
    instead of streaming it. Same contract as the H3 twin."""
    return plan_component_windows(
        usable_gb, range(int(n_blocks)), n_blocks, dict(comp_gb),
        overhead_gb=_K2FT_NF4_OVERHEAD_GB if nf4 else _K2FT_OVERHEAD_GB,
        trunk_gb_per_block=krea2_trunk_gb_per_block(nf4),
        slots_gb=_K2FT_STREAM_SLOTS_GB, allow_stream=allow_stream)


def is_component_spec(spec) -> bool:
    """True when a rotation spec lists component windows rather than block indices.

    Component entries are prefix strings ("mlp.fc1") or (prefix, lo, hi) depth-split
    tuples; block-mode entries are plain ints. Shared by every _targets implementation
    so the two encodings can never drift apart."""
    return bool(spec) and isinstance(spec[0], (str, tuple))


def component_entry_matches(entry, lname: str, block_idx: int) -> bool:
    """Does a component window entry claim this (linear name, block index)?

    A bare prefix claims the Linear at every depth; a (prefix, lo, hi) tuple claims it
    only for blocks lo..hi inclusive — the depth-split that lets a fat window (fc1 is
    15.4 GB of bf16 across 50 H3 blocks) fit a smaller card in slices."""
    if isinstance(entry, str):
        return lname.startswith(entry)
    prefix, lo, hi = entry
    return lname.startswith(prefix) and int(lo) <= int(block_idx) <= int(hi)


def is_rotatable_linear(m: nn.Module) -> bool:
    """A frozen, quantized Linear the rotator can swap to bf16 and back.

    Two storage schemes qualify. fp8 keeps the weight in `.weight` with a companion
    `scale_weight`; NF4 (modules/nf4.py) empties `.weight` to a 0-element tensor and
    parks the real bytes in `_nf4_packed` / `_nf4_state` behind a patched forward.
    Both are frozen and both are reconstructible from the bf16 master, which is all
    rotation needs — so both rotate, and `_activate_targets`/`_deactivate_targets`
    branch per module rather than per model (a mixed model is handled for free)."""
    return isinstance(m, nn.Linear) and (hasattr(m, "scale_weight")
                                         or getattr(m, "_is_nf4", False))


def _linears_with_scale(module: nn.Module) -> List[tuple]:
    """(qualified_name, linear) for every rotatable quantized Linear under `module`."""
    return [(name, m) for name, m in module.named_modules() if is_rotatable_linear(m)]


def _all_linears(module: nn.Module) -> List[tuple]:
    """(qualified_name, linear) for every Linear, quantized or not.

    Needed for always-on modules: the training fp8 pass only targets "blocks." and excludes
    txtfusion, so those Linears are plain bf16 and have no scale_weight to filter on.
    """
    return [(n, m) for n, m in module.named_modules() if isinstance(m, nn.Linear)]


class BlockRotator:
    """Swaps DiT blocks between fp8-frozen and bf16-trainable, in place.

    `master` maps the model-level weight key -> CPU bf16 tensor. It is the
    authoritative copy: activation reads from it, deactivation writes to it.
    """

    def __init__(self, blocks: nn.ModuleList, master: Dict[str, torch.Tensor],
                 key_prefix: str = "blocks", device: str = "cuda",
                 fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                 quantization_mode: str = "block", block_size: int = 64):
        self.blocks = blocks
        self.master = master
        self.key_prefix = key_prefix
        self.device = device
        self.fp8_dtype = fp8_dtype
        self.quantization_mode = quantization_mode
        self.block_size = block_size
        self.active: List[int] = []
        self._patched_forward = {}      # id(linear) -> bound fp8 forward, for restore
        # (key_prefix, module) pairs held trainable for the WHOLE run, never rotated out.
        self.always: List[tuple] = []

    def _key(self, block_idx: int, linear_name: str) -> str:
        return f"{self.key_prefix}.{block_idx}.{linear_name}.weight"

    # ---- always-on modules (e.g. txtfusion) ----
    def activate_always(self, prefix: str, module: nn.Module) -> int:
        """Make every patched Linear under `module` trainable for the entire run.

        txtfusion lives outside dit.blocks, so rotation would never reach it — yet it's the
        stack that fuses the text embeddings, i.e. where prompt-to-concept binding is shaped.
        Leaving it frozen would cap exactly the behaviour a fine-tune is usually after. It's
        small (a 3072-wide text path vs the 6144-wide main blocks) so holding it trainable
        throughout costs little.
        """
        n_swapped = n_direct = 0
        for lname, lin in _all_linears(module):
            key = f"{prefix}.{lname}.weight"
            if hasattr(lin, "scale_weight"):
                # fp8-quantized: same swap as a rotating block, sourced from the master.
                w = self.master.get(key)
                if w is None:
                    logger.warning("[rotation] no master weight for %s — leaving frozen", key)
                    continue
                self._patched_forward[id(lin)] = lin.__dict__.pop("forward", None)
                lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16), requires_grad=True)
                n_swapped += 1
            else:
                # Already bf16 — the training fp8 pass targets "blocks." and excludes
                # txtfusion, so these are the real weights and just need unfreezing.
                lin.weight.requires_grad_(True)
                n_direct += 1
            if lin.bias is not None:
                lin.bias.requires_grad_(True)
        self.always.append((prefix, module))
        logger.info("[rotation] always-on: %s (%d Linears trainable for the whole run"
                    "%s)", prefix, n_swapped + n_direct,
                    f"; {n_swapped} de-quantized, {n_direct} already bf16" if n_swapped else "")
        return n_swapped + n_direct

    # ---- targets ----
    def _targets(self, spec) -> List[tuple]:
        """(key, linear) pairs for a window spec.

        spec is either block indices (block mode) or component window entries (component
        mode) — a bare prefix such as "attn"/"mlp" spans EVERY block, a (prefix, lo, hi)
        tuple spans only that depth slice (the split that fits fat windows on small cards).
        """
        out = []
        if is_component_spec(spec):
            for bi, block in enumerate(self.blocks):
                for lname, lin in _linears_with_scale(block):
                    if any(component_entry_matches(c, lname, bi) for c in spec):
                        out.append((self._key(bi, lname), lin))
        else:
            for bi in spec:
                for lname, lin in _linears_with_scale(self.blocks[bi]):
                    out.append((self._key(bi, lname), lin))
        return out

    # ---- activation: fp8 (frozen) -> bf16 (trainable) ----
    def _activate_targets(self, targets) -> int:
        n = 0
        for key, lin in targets:
            w = self.master.get(key)
            if w is None:
                logger.warning("[rotation] no master weight for %s — leaving frozen", key)
                continue
            # Stash the quantized forward (fp8 dequant or NF4 dequant) so deactivate()
            # can put it back verbatim. Popping it exposes the class-level nn.Linear
            # forward, which reads the bf16 .weight we install next.
            self._patched_forward[id(lin)] = lin.__dict__.pop("forward", None)
            lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16), requires_grad=True)
            if getattr(lin, "_is_nf4", False):
                # Drop the 4-bit copy for as long as this Linear trains: it is pure
                # duplication now (the master is the source of truth and deactivate
                # re-encodes from the trained weight), and holding both is what the
                # small-card tiers cannot afford.
                lin._nf4_packed = None
                lin._nf4_state = None
            if lin.bias is not None:
                lin.bias = nn.Parameter(lin.bias.detach().to(torch.bfloat16), requires_grad=True)
            n += 1
        return n

    def _deactivate_targets(self, targets) -> int:
        from fizgig.krea2.fp8_optimization_utils import (
            calculate_fp8_maxval, quantize_weight, fp8_linear_forward_patch,
        )
        max_value = calculate_fp8_maxval(4, 3)
        min_value = -max_value
        n = 0
        for key, lin in targets:
            if key not in self.master:
                continue
            trained = lin.weight.detach()
            # Master is the source of truth — save BEFORE the lossy re-quantize. This is
            # also why the re-encode below can round to NEAREST: the residency copy is
            # only the frozen forward CONTEXT for later windows, never the training
            # signal, and activate() always reads the master back, so nothing compounds
            # across cycles. (H3 needed stochastic rounding for its SAVE, where trained
            # int8 deltas smaller than the grid step rounded home; Krea 2 saves bf16
            # straight from this master, so it has no lossy save to protect.)
            self.master[key] = trained.to("cpu", dtype=torch.bfloat16).clone()
            if getattr(lin, "_is_nf4", False):
                from bitsandbytes.functional import quantize_nf4
                from fizgig.modules.nf4 import nf4_linear_forward_patch
                packed, state = quantize_nf4(trained.contiguous(),
                                             compress_statistics=False)
                lin._nf4_packed = packed
                lin._nf4_state = state
                # apply_nf4_quantization's own convention: keep the Parameter object
                # (module identity and in/out_features stay intact) but empty it.
                lin.weight.data = torch.empty(0, device=packed.device,
                                              dtype=torch.bfloat16)
                lin.weight.requires_grad_(False)
                saved = self._patched_forward.pop(id(lin), None)
                lin.forward = (saved if saved is not None
                               else nf4_linear_forward_patch.__get__(lin, type(lin)))
            else:
                q, scale = quantize_weight(key, trained.float(), self.fp8_dtype,
                                           max_value, min_value,
                                           quantization_mode=self.quantization_mode,
                                           block_size=self.block_size)
                lin.weight = nn.Parameter(q.to(self.device), requires_grad=False)
                sw = scale.to(self.device, dtype=lin.scale_weight.dtype).reshape(lin.scale_weight.shape)
                lin.scale_weight.copy_(sw)
                saved = self._patched_forward.pop(id(lin), None)
                if saved is not None:
                    lin.forward = saved
                else:
                    def _fwd(self_, x, _p=fp8_linear_forward_patch):
                        return _p(self_, x, False, None)
                    lin.forward = _fwd.__get__(lin, type(lin))
            if lin.bias is not None:
                lin.bias.requires_grad_(False)
            # RELEASE THE ORPHAN — see the H3 twin in rotation_ft.py for the measurement.
            # Rebinding lin.weight above does not free the old bf16: a C++-side autograd
            # referrer keeps its storage alive and the whole outgoing window survives into
            # the next epoch, so a rotation costs two windows instead of one. Nothing
            # legitimate reads it again — the master holds a CPU clone and both encoders
            # take their own copy (fp8 via .float(), NF4 by packing to uint8) — and
            # resize_(0) leaves any phantom holder a zero-byte husk, exactly as
            # park_dit_partial does.
            # GUARDED against the mirror hazard: freeing a storage the module still reads
            # would hand it a zero-byte weight and kill the next forward. That both
            # encoders copy is THEIR property, not ours, so check it rather than assume
            # it — the NF4 bytes live in _nf4_packed, not in the emptied .weight.
            _orphan = trained.untyped_storage()
            del trained
            _live = [lin.weight, getattr(lin, "_nf4_packed", None)]
            if all(t is None or t.untyped_storage().data_ptr() != _orphan.data_ptr()
                   for t in _live):
                try:
                    _orphan.resize_(0)
                except Exception:
                    pass
            n += 1
        return n

    def activate(self, spec) -> int:
        spec = list(spec)
        n = self._activate_targets(self._targets(spec))
        # Keep self.active in step — trainable_params() and master_state_dict() read it.
        # Component entries (strings / (prefix, lo, hi) tuples) keep their order; only
        # block-index specs get sorted.
        merged = list(self.active) + [x for x in spec if x not in self.active]
        self.active = merged if is_component_spec(merged) else sorted(merged)
        logger.info("[rotation] activated %s (%d Linears now trainable)", spec, n)
        return n

    def deactivate(self, spec) -> int:
        spec = list(spec)
        n = self._deactivate_targets(self._targets(spec))
        self.active = [x for x in self.active if x not in spec]
        logger.info("[rotation] deactivated %s (%d Linears re-quantized)", spec, n)
        return n

    def rotate_to(self, spec) -> None:
        """Make exactly `spec` trainable, deactivating whatever else is active."""
        spec = list(spec)
        if spec == list(self.active):
            return
        if self.active:
            self._deactivate_targets(self._targets(list(self.active)))
            # Hand the freed blocks back before allocating the next window: without this the
            # old window's bf16 tensors sit in the allocator's reserve and the new window
            # OOMs against its own predecessor.
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        self._activate_targets(self._targets(spec))
        self.active = spec
        logger.info("[rotation] window -> %s", spec)

    def trainable_params(self) -> List[nn.Parameter]:
        """Active-window params PLUS the always-on modules. Rebuilt each rotation, so the
        always-on params must be re-included every time or they'd silently stop updating."""
        params, seen = [], set()
        for _key, lin in self._targets(list(self.active)):
            for p in (lin.weight, lin.bias):
                if p is not None and p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        for _prefix, module in self.always:
            for p in module.parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def master_state_dict(self) -> Dict[str, torch.Tensor]:
        """The bf16 master, with currently-active blocks and always-on modules flushed in."""
        out = dict(self.master)
        for key, lin in self._targets(list(self.active)):
            if key in out:
                out[key] = lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()
        for prefix, module in self.always:
            # Export unconditionally: unquantized always-on Linears were never in the master
            # (they live on the GPU in bf16 from the start), but they HAVE been trained.
            for lname, lin in _all_linears(module):
                out[f"{prefix}.{lname}.weight"] = (
                    lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone())
        return out


# ---------------------------------------------------------------------------
# Rotation-aware block swap
# ---------------------------------------------------------------------------

class RotationOffloader:
    """Keeps the trainable window pinned on GPU and streams every other block from CPU.

    The stock offloader keeps a contiguous PREFIX of blocks resident and swaps the suffix,
    pairing each eviction with a matching load so residency stays constant. Rotation needs
    the opposite: an arbitrary resident set (the active window) that MOVES every rotation.
    Rather than rework that pairing, this streams non-resident blocks just-in-time.

    Implements the same small interface the DiT's forward expects, so it can be dropped in
    as `dit.offloader` with no changes to model.py.

    Gradient checkpointing recomputes a block's forward *inside* its backward, so blocks are
    pulled back to the GPU on a backward PRE-hook (before the recompute) and evicted on the
    post-hook — evicting on the forward pass alone would leave the recompute with weights on
    the wrong device.
    """

    def __init__(self, blocks: nn.ModuleList, device: torch.device, resident: Iterable[int],
                 debug: bool = False, prefetch: bool = True):
        self.blocks = blocks
        self.device = torch.device(device)
        self.cpu = torch.device("cpu")
        self.resident = set(int(i) for i in resident)
        self.forward_only = False
        self.debug = debug
        self.moves = 0                      # instrumentation: transfers per step
        self._handles = []
        # Async plumbing: one worker thread issuing copies on a dedicated CUDA stream, so a
        # block's transfer overlaps the previous block's compute instead of stalling on it.
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.prefetch = bool(prefetch and self.cuda)
        self._pool = ThreadPoolExecutor(max_workers=1) if self.prefetch else None
        self._stream = torch.cuda.Stream(device=self.device) if self.prefetch else None
        self._futures = {}                  # block idx -> Future[cuda.Event | None]
        self._register_hooks()

    # -- placement helpers -------------------------------------------------
    def _to(self, idx: int, dev: torch.device):
        from fizgig.krea2.offloading import weighs_to_device
        weighs_to_device(self.blocks[idx], dev)
        # NF4 blocks keep their real bytes in `_nf4_packed` / `_nf4_state`, which are
        # plain attributes that weighs_to_device (and nn.Module.to) cannot see — so a
        # block would "move" while its 4-bit weight stayed put, and the dequant forward
        # then met mat2 on the wrong device (field: RuntimeError at the first rotation
        # boundary of an NF4 fine-tune). move_nf4_to_device exists for exactly this and
        # no-ops on an fp8 block.
        from fizgig.modules.nf4 import move_nf4_to_device
        move_nf4_to_device(self.blocks[idx], dev)
        self.moves += 1

    def _move_task(self, idx: int, dev: torch.device, after):
        """Runs on the worker thread, issuing the copy on the transfer stream."""
        torch.cuda.set_device(self.device.index if self.device.index is not None
                              else torch.cuda.current_device())
        with torch.cuda.stream(self._stream):
            if after is not None:
                # Don't yank weights out from under kernels that are still queued.
                self._stream.wait_event(after)
            self._to(idx, dev)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        return ev

    def _submit(self, idx: int, dev: torch.device, after=None):
        # Track the DIRECTION, not just that something is in flight: an in-flight eviction
        # must never be mistaken for a completed load (that race put weights on CPU while the
        # forward was still running, and only showed up as a device mismatch deep in fp8).
        pending = self._futures.get(idx)
        if pending is not None:
            if pending[0] == dev:
                return                  # same direction already queued
            self._await(idx)            # opposite direction: settle it first
        self._futures[idx] = (dev, self._pool.submit(self._move_task, idx, dev, after))

    def _await(self, idx: int):
        """Settle any in-flight move for `idx`. Returns the device it moved to, or None."""
        pending = self._futures.pop(idx, None)
        if pending is None:
            return None
        dev, fut = pending
        ev = fut.result()
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        return dev

    def _ensure_gpu(self, idx: int):
        if idx in self.resident:
            return
        landed = self._await(idx)
        if landed is not None and landed.type == self.device.type:
            return                      # a prefetch already put it on the GPU
        self._to(idx, self.device)

    def _evict(self, idx: int):
        if idx in self.resident:
            return
        if self.prefetch:
            # Queue the eviction behind the compute that just used this block.
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream())
            self._submit(idx, self.cpu, after=done)
        else:
            self._to(idx, self.cpu)

    def _prefetch_next(self, idx: int):
        if not self.prefetch or idx < 0 or idx >= len(self.blocks) or idx in self.resident:
            return
        self._submit(idx, self.device)

    # -- rotation ----------------------------------------------------------
    def set_resident(self, resident: Iterable[int]):
        """Re-pin after a rotation: the incoming window goes to GPU and stays, the outgoing
        window rejoins the streamed pool."""
        for idx in list(self._futures):     # settle everything before re-planning
            self._await(idx)
        new = set(int(i) for i in resident)
        # EVICT BEFORE LOADING. The reverse order holds the outgoing and incoming
        # windows on the card at once — a transient of one full window, which is what
        # tips a tight card over (the H3 twin was fixed the same way, 27 Aug). Blocks in
        # both sets appear in neither difference, so they are never touched.
        for idx in sorted(self.resident - new):
            self._to(idx, self.cpu)
        for idx in sorted(new - self.resident):
            self._to(idx, self.device)
        self.resident = new
        logger.info("[rotation-swap] resident blocks now %s (others stream from CPU)", sorted(new))

    def view(self, start: int, count: int) -> "OffloaderView":
        """The offloader for one of the model's block lists when `blocks` is several of them end to end (Klein:
        8 double then 24 single blocks, each list swapped by its own offloader counting from 0)."""
        return OffloaderView(self, int(start), int(count))

    def _place(self, idxs):
        for idx in list(self._futures):
            self._await(idx)
        for i in idxs:
            self._to(i, self.device if i in self.resident else self.cpu)
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # Warm the first streamed block so step 0 isn't a stall.
        if idxs:
            self._prefetch_next(idxs[0] if idxs[0] not in self.resident else idxs[0] + 1)

    # -- interface the DiT forward calls -----------------------------------
    def prepare_block_devices_before_forward(self, blocks):
        self._place(range(len(blocks)))

    def wait_for_block(self, index: int):
        self._ensure_gpu(index)
        self._prefetch_next(index + 1)      # overlap the next load with this block's compute

    def submit_move_blocks_forward(self, blocks, index: int):
        self._evict(index)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = bool(forward_only)

    # -- backward ----------------------------------------------------------
    def _register_hooks(self):
        for i, block in enumerate(self.blocks):
            def pre_hook(module, grad_output, _i=i):
                self._ensure_gpu(_i)
                self._prefetch_next(_i - 1)   # backward walks blocks in reverse
                return None

            def post_hook(module, grad_input, grad_output, _i=i):
                self._evict(_i)
                return None

            self._handles.append(block.register_full_backward_pre_hook(pre_hook))
            self._handles.append(block.register_full_backward_hook(post_hook))

    def remove(self):
        for idx in list(self._futures):
            self._await(idx)
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True)


class OffloaderView:
    """One block list inside a RotationOffloader that streams several lists end to end: the same interface the
    forward calls, with the list's own block index shifted to its place in the streamer. Prefetch runs on across the
    boundary (the last double block warms the first single block)."""

    def __init__(self, owner: RotationOffloader, start: int, count: int):
        self.owner, self.start, self.count = owner, start, count

    def prepare_block_devices_before_forward(self, blocks):
        self.owner._place(range(self.start, self.start + len(blocks)))

    def wait_for_block(self, index: int):
        self.owner.wait_for_block(self.start + index)

    def submit_move_blocks_forward(self, blocks, index: int):
        self.owner.submit_move_blocks_forward(None, self.start + index)

    def set_forward_only(self, forward_only: bool):
        self.owner.set_forward_only(forward_only)
