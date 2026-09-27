"""The standard layer's workbench engine: Repair Studio (and later the Explorer and Royale) for any described family.

One engine over a family's driver (model code) and the family LoRA layer (adapters, block controls, bake). It speaks
the same protocol as the old per-family engines (repair_studio.engine / krea2_engine / h3_engine), so the tabs drive
it unchanged: ensure_pipeline, load_primary / load_donor / unload_donor, swap_primary_weights, apply_state,
generate_preview, generate_baseline, request_cancel / clear_cancel, reset, plus the primary_* / donor_* attributes.

Previews run with the family's speed LoRA (the description's preview_speed_lora) when its file is set and the user
picks it, else with the family's default sampling. Memory follows the trainer's rules: the text encoder loads only
when a prompt changes, and the DiT parks on CPU while it runs whenever both would not fit; on cards under 20 GB the
DiT also parks for the VAE decode.

A LoRA's load strength (state.primary_scale / donor_scale) scales the whole file and every block slider is relative
to it; the bake folds the sliders in but never the load strength, so the saved file is used at that strength.
"""
import gc
import json
import logging
import os
import threading
from typing import Optional

import torch

from fizgig.families.driver import FamilyDriver
from fizgig.families.lora import FamilyLoRA

logger = logging.getLogger(__name__)

PRIMARY, DONOR, SPEED = "primary", "donor", "speed_lora"


class RenderCancelled(Exception):
    """A render aborted at a step boundary by request_cancel (a newer edit wants the engine)."""


class _Loaded:
    is_loaded = True


def _slerp(t, a, b):
    """Spherical interpolation of two noise tensors (keeps the norm, so a travel frame never goes mushy)."""
    af, bf = a.flatten().float(), b.flatten().float()
    na, nb = af.norm(), bf.norm()
    if na.item() == 0 or nb.item() == 0:
        return torch.lerp(a.float(), b.float(), t)
    dot = torch.dot(af / na, bf / nb).clamp(-1.0, 1.0)
    om = torch.acos(dot)
    so = torch.sin(om)
    if so.item() < 1e-6:
        return torch.lerp(a.float(), b.float(), t)
    return ((torch.sin((1 - t) * om) / so) * a.float() + (torch.sin(t * om) / so) * b.float())


def _blend(a, b, t, mode):
    af, bf = a.float(), b.float()
    if mode in (None, "lerp"):
        return torch.lerp(af, bf, t).to(a.dtype)
    eps = 1e-6
    na, nb = af.norm(dim=-1, keepdim=True), bf.norm(dim=-1, keepdim=True)
    target = (1 - t) * na + t * nb
    if mode == "norm":
        out = torch.lerp(af, bf, t)
        return (out * (target / out.norm(dim=-1, keepdim=True).clamp_min(eps))).to(a.dtype)
    ua, ub = af / na.clamp_min(eps), bf / nb.clamp_min(eps)
    dot = (ua * ub).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    om = torch.acos(dot)
    so = torch.sin(om)
    arc = (torch.sin((1 - t) * om) / so) * ua + (torch.sin(t * om) / so) * ub
    d = torch.where(so < 1e-4, torch.lerp(ua, ub, t), arc)
    return (d * target).to(a.dtype)


def _free_vram_gb():
    try:
        from fizgig.families.quant import free_vram_gb
        return free_vram_gb()
    except Exception:
        return 0.0


class WorkbenchEngine:
    def __init__(self, description):
        self.desc = description
        self.driver = description.load_driver()
        self.pipeline = None
        self.dit = self.vae = self.net = None
        self.te_path = None
        self.device = "cuda"
        self.speed = None                   # the SpeedLoRA previews use, or None (default sampling)
        self.lowmem = False

        self.primary_network = None         # the FamilyLoRA once a primary is attached (the tabs test for None)
        self.donor_network = None
        self.primary_path = self.donor_path = None
        self.primary_block_ids, self.donor_block_ids = set(), set()
        self.primary_hash = None

        self._prompt_cache = {}             # (prompt,) -> conditioning dict on CPU
        self._cancel_event = threading.Event()
        self.on_step = None                 # (done, total) per denoising step, from the render thread
        self._baseline_key = self._baseline_img = None
        self._turbo_enabled = False         # no activation cache: the speed LoRA already makes previews short
        self._last_frame_latent = None      # Royale's shared workers read it; there is no latent chaining here

    # ---- the block map ------------------------------------------------------------------------------
    def block_groups(self):
        return self.driver.block_map(self.dit)

    def block_ids(self):
        return [b.id for g in self.block_groups() for b in g.blocks]

    def default_state(self, width=None, height=None):
        """A SliderState over this family's blocks (every slider at its default)."""
        from fizgig.repair_studio.state import BlockState, SliderState
        s = SliderState(blocks={bid: BlockState() for bid in self.block_ids()})
        if width:
            s.preview_width, s.preview_height = int(width), int(height or width)
        return s

    # ---- models -------------------------------------------------------------------------------------
    def ensure_pipeline(self, dit_path, vae_path, text_encoder_path, speed_lora_path="", device="cuda",
                        lowmem=None, precision="auto", blocks_to_swap=0, **_ignored):
        """Load the DiT (resident) and the VAE once; the text encoder loads per new prompt. speed_lora_path: the
        family's speed LoRA file, attached unmerged and used for every preview ("" = default sampling).
        precision: "auto" = bf16 with 20 GB+ free, else INT8 (when the family offers it); blocks_to_swap streams
        blocks forward-only (previews never backprop)."""
        if self.pipeline is not None:
            return
        from fizgig.families.train import _small_card_previews
        self.device = device
        self.te_path = text_encoder_path
        self.lowmem = _small_card_previews() if lowmem is None else bool(lowmem)
        from fizgig.families import quant
        if precision == "auto":
            precision = "int8" if ("int8" in self.desc.precisions and _free_vram_gb() < 20.0) else "bf16"
        elif precision not in self.desc.precisions:
            precision = "bf16"
        self.dit, self.swapped = quant.load_base(self.driver, dit_path, device, precision, blocks_to_swap,
                                                 supports_backward=False)
        if self.swapped:
            self.driver.block_swap_mode(self.dit, inference=True)
        self.precision = precision
        self.vae = self.driver.load_vae(vae_path, "cpu" if self.lowmem else device)
        self.net = FamilyLoRA(self.dit, self.driver, device=device)
        sp = self.desc.preview_speed()
        if speed_lora_path and sp is not None and os.path.exists(speed_lora_path):
            n = self.net.add_file(speed_lora_path, SPEED, sp.strength)
            self.net.move_adapter(SPEED, device)
            self.speed = sp
            logger.info("%s workbench: speed LoRA %s on %d Linears", self.desc.display_name, sp.name, n)
        self.pipeline = _Loaded()
        logger.info("%s workbench ready (%s, block swap %d, lowmem=%s)", self.desc.display_name, precision,
                    self.swapped, self.lowmem)

    def _attach(self, path, name, strength=1.0):
        n = self.net.add_file(path, name, strength)
        if n == 0:
            self.net.remove(name)
            raise ValueError(f"{os.path.basename(path)} adapts nothing in {self.desc.display_name} "
                             "(a LoRA for another model?)")
        self.net.move_adapter(name, self.device)
        return n

    def load_primary(self, path):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded; call ensure_pipeline() first.")
        if self.primary_network is not None:
            raise RuntimeError("Primary already loaded — call reset() to swap.")
        n = self._attach(path, PRIMARY)
        self.primary_network = self.net
        self.primary_path = path
        self.primary_block_ids = self.net.adapter_blocks(PRIMARY)
        self.primary_hash = self._hash(path)
        self._invalidate_baseline_cache()
        logger.info("%s primary: %s (%d Linears, %d blocks)", self.desc.display_name, path, n,
                    len(self.primary_block_ids))

    def swap_primary_weights(self, path) -> bool:
        """Another checkpoint of the same run in place (Royale's epoch scrub). False = the caller should reset()
        and load_primary() instead."""
        if self.primary_network is None:
            raise RuntimeError("No primary loaded; call load_primary() first.")
        try:
            n = self.net.swap_file(PRIMARY, path)
        except Exception:
            logger.exception("swap_primary_weights: %s", path)
            return False
        if n == 0:
            return False
        self.net.move_adapter(PRIMARY, self.device)
        self.primary_path = path
        self.primary_block_ids = self.net.adapter_blocks(PRIMARY)
        self.primary_hash = self._hash(path)
        self._invalidate_baseline_cache()
        return True

    def load_donor(self, path):
        if self.primary_network is None:
            raise RuntimeError("Load primary LoRA before donor.")
        if self.donor_network is not None:
            raise RuntimeError("Donor already loaded — unload_donor() or reset() first.")
        self._attach(path, DONOR)
        # The donor comes in through its block sliders only (strength 0 by default); anything it adapts outside
        # the block map stays off, as on the other engines.
        self.net.set_outside(DONOR, False)
        self.donor_network = self.net
        self.donor_path = path
        self.donor_block_ids = self.net.adapter_blocks(DONOR)

    def unload_donor(self):
        if self.donor_network is not None:
            self.net.remove(DONOR)
            self.donor_network = None
            self.donor_path = None
            self.donor_block_ids = set()

    @staticmethod
    def _hash(path):
        try:
            from fizgig.profiler.visualize import compute_lora_hash
            return compute_lora_hash(path)
        except Exception:
            return None

    # ---- slider state -------------------------------------------------------------------------------
    def apply_state(self, state):
        """Per-block strength / on-off and each LoRA's load strength, live (no reload)."""
        if self.primary_network is None:
            return
        for name, who in ((PRIMARY, "primary"), (DONOR, "donor")):
            if not self.net.has(name):
                continue
            self.net.set_blocks(
                name, mult={b: float(getattr(bs, f"{who}_strength")) for b, bs in state.blocks.items()},
                enabled={b: bool(getattr(bs, f"{who}_enabled")) for b, bs in state.blocks.items()})
            self.net.set_strength(name, float(getattr(state, f"{who}_scale", 1.0)))

    def mark_blocks_changed(self, blocks):
        pass                                # every render is a full forward (no activation cache)

    # ---- cancellation -------------------------------------------------------------------------------
    def request_cancel(self):
        self._cancel_event.set()

    def clear_cancel(self):
        self._cancel_event.clear()

    # ---- rendering ----------------------------------------------------------------------------------
    def _park_dit(self, where):
        if self.dit is not None and not getattr(self, "swapped", 0):   # a swapped DiT keeps its streaming layout
            from fizgig.families import quant
            quant.move(self.dit, where)
            if where == "cpu":
                gc.collect()
                torch.cuda.empty_cache()

    def encode(self, prompts):
        """Conditioning for each prompt (CPU), through the family's text encoder, loaded for the call and freed.
        The DiT parks on CPU while the encoder runs when both would not fit."""
        need = [p for p in prompts if (p,) not in self._prompt_cache]
        if need:
            te_gb = os.path.getsize(self.te_path) / 1024 ** 3 if os.path.exists(self.te_path) else 0.0
            park = self.lowmem or _free_vram_gb() < te_gb + 2.0
            if park:
                self._park_dit("cpu")
            try:
                te = self.driver.load_text_encoder(self.te_path, self.device)
                try:
                    for p, c in zip(need, self.driver.encode_text(te, list(need))):
                        self._prompt_cache[(p,)] = c
                finally:
                    self.driver.unload_text_encoder(te)
                    del te
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                if park:
                    self._park_dit(self.device)
        return [self._prompt_cache[(p,)] for p in prompts]

    def _cond_to_device(self, cond):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in cond.items()}

    def sampling(self):
        """(steps, cfg, sigmas, options) previews use."""
        if self.speed is not None:
            s = self.speed.settings
            return s.steps, s.cfg, s.sigmas, s.options
        s = self.desc.default_sampling()
        return s.steps, s.cfg, s.sigmas, s.options

    @torch.no_grad()
    def render(self, cond, width, height, seed, *, steps=None, noise=None):
        """One image from conditioning with the adapters as currently set."""
        d_steps, cfg, sigmas, options = self.sampling()
        steps = int(steps or d_steps)

        def _step(done, total):
            cb = self.on_step
            if cb is not None:
                try:
                    cb(done, total)
                except Exception:
                    pass
            if self._cancel_event.is_set():
                raise RenderCancelled()

        lat = self.driver.generate(self.dit, self._cond_to_device(cond), width, height, steps=steps, seed=int(seed),
                                   cfg=cfg, sigmas=sigmas, options=options, noise=noise, on_step=_step)
        if self.lowmem:
            self._park_dit("cpu")
            self.vae.to(self.device)
        try:
            return self.driver.decode(self.vae, lat, width, height)
        finally:
            if self.lowmem:
                self.vae.to("cpu")
                self._park_dit(self.device)

    def generate_preview(self, state, *, seed=None, prompt=None, width=None, height=None, steps=None,
                         seed_b=None, travel_t=0.0, override_ctx=None, override_neg_ctx=None,
                         prev_latent=None, prev_latent_strength=1.0):
        """The tabs' render call (signature shared with the old engines; the Klein-only reference-latent and
        negative arguments are accepted and ignored). seed_b / travel_t: seed travel by noise slerp.
        override_ctx: precomputed conditioning (prompt travel)."""
        self.apply_state(state)
        seed = state.seed if seed is None else seed
        width = int(width or state.preview_width)
        height = int(height or state.preview_height)
        cond = override_ctx if override_ctx is not None else self.encode([prompt if prompt is not None
                                                                          else state.prompt])[0]
        noise = None
        if seed_b is not None:
            noise = _slerp(float(travel_t or 0.0), self.driver.initial_noise(seed, width, height),
                           self.driver.initial_noise(seed_b, width, height))
        return self.render(cond, width, height, seed, steps=steps, noise=noise)

    def generate_baseline(self, state):
        """The primary with every slider at 1.0 (at its load strength), donor off. Cached until the prompt, seed,
        size or load strength changes."""
        key = (self.primary_path, state.seed, state.prompt, state.preview_width, state.preview_height,
               round(float(getattr(state, "primary_scale", 1.0)), 4))
        if self._baseline_key == key and self._baseline_img is not None:
            return self._baseline_img
        base = self.default_state(state.preview_width, state.preview_height)
        base.seed, base.prompt = state.seed, state.prompt
        base.primary_scale = float(getattr(state, "primary_scale", 1.0))
        img = self.generate_preview(base)
        self._baseline_key, self._baseline_img = key, img
        return img

    def _invalidate_baseline_cache(self):
        self._baseline_key = self._baseline_img = None

    def _invalidate_activation_cache(self):
        pass

    # ---- prompt travel (Royale) -------------------------------------------------------------------------
    @property
    def supports_prompt_travel(self):
        return type(self.driver).pad_conditioning is not FamilyDriver.pad_conditioning

    def encode_travel_prompts(self, prompts):
        """Waypoint conditioning, padded to one shape by the driver. Returns (waypoints, None) - the second slot is
        the old engines' negative, unused here."""
        return self.driver.pad_conditioning(self.encode(list(prompts))), None

    @staticmethod
    def interp_waypoints(vecs, t, mode="lerp"):
        """Piecewise blend across the waypoint dicts (t 0..1 walks the whole chain): float tensors by lerp / norm /
        slerp along the feature axis; boolean masks become weights (a token only one side has fades in or out), so
        each endpoint is exactly its own prompt and nothing switches on mid-travel."""
        if len(vecs) == 1:
            return vecs[0]
        t = min(max(float(t), 0.0), 1.0)
        pos = t * (len(vecs) - 1)
        i = min(int(pos), len(vecs) - 2)
        local = pos - i
        out = {}
        for k, a in vecs[i].items():
            b = vecs[i + 1][k]
            if not torch.is_tensor(a):
                out[k] = a
            elif a.dtype == torch.bool:
                out[k] = a if local == 0.0 else torch.lerp(a.float(), b.float(), local)
            else:
                out[k] = _blend(a, b, local, mode)
        return out

    # ---- bake ---------------------------------------------------------------------------------------
    def save_repaired(self, out_path, state, include_donor=True):
        """Write the primary (and the donor's enabled blocks) as one LoRA in the family's format: block sliders
        folded in, load strengths NOT (the file is used at them, as previewed). Returns the summary dict the
        Repair Studio's save dialog reports."""
        from safetensors import safe_open
        from safetensors.torch import save_file
        use_donor = include_donor and self.net.has(DONOR)
        flat = state.copy()
        flat.primary_scale = flat.donor_scale = 1.0
        self.apply_state(flat)
        try:
            names = [PRIMARY] + ([DONOR] if use_donor else [])
            sd, ranks = self.net.bake(names)
        finally:
            self.apply_state(state)
        try:
            with safe_open(self.primary_path, "pt") as f:
                metadata = {str(k): str(v) for k, v in (f.metadata() or {}).items()}
        except Exception:
            metadata = {}
        for stale in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"):
            metadata.pop(stale, None)
        if ranks:
            metadata["ss_network_dim"] = str(max(ranks.values()))
            metadata["ss_network_alpha"] = str(float(max(ranks.values())))
        metadata["ss_repair_studio_config"] = json.dumps(state.to_json(), separators=(",", ":"))
        blended = set()
        if use_donor:
            metadata["ss_repair_studio_donor_path"] = os.path.basename(self.donor_path)
            blended = {b for b, bs in state.blocks.items()
                       if b in self.donor_block_ids and bs.donor_enabled and bs.donor_strength != 0}
            combined = {m: r for m, r in ranks.items() if self.driver.block_of(m) in blended}
            if combined:
                metadata["ss_repair_studio_combined_ranks"] = json.dumps(dict(sorted(combined.items())),
                                                                         separators=(",", ":"))
        save_file(sd, out_path, metadata=metadata)
        kept = {self.driver.block_of(m) for m in ranks}
        dropped = sorted((self.primary_block_ids - kept) - blended)
        rescaled = sorted(b for b, bs in state.blocks.items()
                          if b in self.primary_block_ids and b not in dropped and bs.primary_enabled
                          and abs(bs.primary_strength - 1.0) > 1e-9)
        return {"dropped_blocks": dropped, "rescaled_blocks": rescaled, "blended_blocks": sorted(blended),
                "keys_in": 3 * len(self.net._frozen[PRIMARY]["alpha_rank"]), "keys_out": len(sd),
                "donor_path": self.donor_path if use_donor else None, "format_out": "standard",
                "lycoris_converted": 0}

    # ---- teardown -----------------------------------------------------------------------------------
    def reset(self):
        from fizgig.utils.device import release_module_tensors
        for m in (self.dit, self.vae):
            if m is not None:
                try:
                    release_module_tensors(m)
                except Exception:
                    pass
        self.dit = self.vae = self.net = None
        self.pipeline = None
        self.speed = None
        self.primary_network = self.donor_network = None
        self.primary_path = self.donor_path = None
        self.primary_block_ids, self.donor_block_ids = set(), set()
        self.primary_hash = None
        self._prompt_cache = {}
        self._invalidate_baseline_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            from fizgig.utils.device import flush_reserved_vram, report_cuda_leak
            report_cuda_leak(f"{self.desc.key}-workbench-reset")
            flush_reserved_vram(f"{self.desc.key}-workbench-reset")
        except Exception:
            pass
