"""FamilyDriver: the one interface a new model family implements (the "driver" behind Fizgig's standard layer).

Fizgig's generic code - caching, training, previews, the LoRA layer, the fetcher, the GUI path and (later) the
workbench tools - talks to a family ONLY through its FamilyDescription (facts) and its FamilyDriver (model code).
A new family is: a description + a driver + its model package. Nothing else in Fizgig changes to add it.

Klein, Krea 2 and MiniMax H3 are not drivers; they keep their own code paths until this layer is proven.

Conventions every driver follows:
* Latents are (C, h, w) tensors in the family's normalised space (what the DiT is trained on).
* Conditioning is a dict of tensors per caption, keys chosen by the driver; the generic cache stores it as-is
  and hands the same dict back (batched, leading dim 1) to training_loss / generate.
* Images in and out are uint8 RGB numpy arrays (H, W, 3) / PIL images.
"""
from __future__ import annotations

from typing import Optional


class FamilyDriver:
    """Subclass per family. `description` is the family's FamilyDescription."""

    description = None

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path: str, device, precision: str = "bf16"):
        """The diffusion transformer, frozen, ready for LoRA wrapping (gradient checkpointing available)."""
        raise NotImplementedError

    def load_vae(self, path: str, device):
        raise NotImplementedError

    def load_text_encoder(self, path: str, device):
        raise NotImplementedError

    def unload_text_encoder(self, te) -> None:
        """Free the encoder's VRAM (caching and preview-prompt encoding load it, then drop it)."""
        raise NotImplementedError

    def enable_gradient_checkpointing(self, dit, on: bool = True) -> None:
        raise NotImplementedError

    # ---- encoding (the generic cache calls these) -----------------------------------------------
    def encode_images(self, vae, images: list) -> list:
        """uint8 (H, W, 3) arrays, all the same size -> list of (C, h, w) latents."""
        raise NotImplementedError

    def encode_text(self, te, captions: list) -> list:
        """captions -> list of conditioning dicts (tensors on CPU)."""
        raise NotImplementedError

    # ---- training -------------------------------------------------------------------------------
    def training_loss(self, dit, latents, cond: dict, generator, *, min_t: float = 0.0, max_t: float = 1.0):
        """One training forward. latents (1, C, h, w) on device, cond = the cached dict (batched).
        Returns (loss tensor, info dict e.g. {"t": 0.63}). Owns the family's noise/target/timestep rules."""
        raise NotImplementedError

    # ---- sampling -------------------------------------------------------------------------------
    def generate(self, dit, cond: dict, width: int, height: int, *, steps: int, seed: int, cfg: float = 1.0,
                 neg_cond: Optional[dict] = None, sigmas=None, options=()):
        """Denoise one image from noise; returns latents in the driver's own layout (fed to decode).
        sigmas / options: an explicit schedule and driver-specific sampler options (e.g. from a speed LoRA's
        SamplingSettings); a driver ignores what it doesn't use."""
        raise NotImplementedError

    def decode(self, vae, latents, width: int, height: int):
        """-> PIL.Image (RGB)."""
        raise NotImplementedError

    # ---- LoRA -----------------------------------------------------------------------------------
    def lora_target_names(self, dit) -> list:
        """Dotted module names (relative to dit) of the Linears a LoRA wraps. Default: every
        '{block_prefix}.{N}.{module}' in the description's LoRA format that exists in the model."""
        d = self.description
        names = {n for n, _ in dit.named_modules()}
        out = []
        for i in range(d.n_blocks):
            for m in d.lora.block_modules:
                n = f"{d.block_prefix}.{i}.{m}"
                if n in names:
                    out.append(n)
        return out

    def block_of(self, module_name: str) -> int:
        """Block index of a wrapped module name (Repair Studio / block targeting)."""
        prefix = self.description.block_prefix + "."
        return int(module_name[len(prefix):].split(".")[0])
