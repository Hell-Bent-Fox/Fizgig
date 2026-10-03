"""MiniMax H3 driver for Fizgig's standard layer (families/driver.py) - first stage of the H3 port.

The old H3 trainer's own code behind the FamilyDriver interface, so the shared cache / train code runs H3 exactly as
it does today:
* caching: the old scripts' cache functions (scripts/minimax_cache_latents.cache_latents / minimax_cache_text
  .cache_text) over the datasets the shared cache loaded - the same minimaxh3 files, byte for byte (cache_stage)
* conditioning: the cached Qwen3-VL-32B layer-50 states, plus a clip's audio rows and a voice item's audio_only flag
  (batch_cond)
* training: minimax/trainer.compute_loss - flow matching at the shift-12 sigma density, audio on its own remapped
  schedule, a voice item's video term left out
* LoRA: the blocks' attention + MLP Linears (AdaLN left out, as --no_train_adaln), block ids h3blk_N / h3_rf_N

Still to come (the doc's §5.7 order): quant tiers + H2D rings, modality routing, TREAD, likeness cut, adapters, two
bases, previews with the clip contract, the workbench wrap, FT.
"""
import torch

from fizgig.families.driver import Block, BlockGroup, FamilyDriver

DTYPE = torch.bfloat16
_BLOCK_MODULES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


class MiniMaxDriver(FamilyDriver):

    loads_quantized = True            # the file is int8 ConvRot: load_dit returns it ready, nothing re-quantises

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.minimax.loader import load_minimax_h3_dit
        return load_minimax_h3_dit(path, device=device, compute_dtype=DTYPE, quantize=True,
                                   base_quant="int8").requires_grad_(False)

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(bool(on))

    # ---- caching: today's H3 cache code, over the datasets the shared cache loaded ------------------
    def cache_stage(self, stage, datasets, args, device, aux):
        import argparse
        common = dict(skip_existing=args.skip_existing, keep_cache=args.keep_cache, num_workers=args.num_workers)
        if stage == "latents":
            from fizgig.scripts.minimax_cache_latents import cache_latents
            cache_latents(argparse.Namespace(vae=args.model, audio_vae=aux.get("audio_vae") or None,
                                             clip_still=aux.get("clip_still") == "1", batch_size=args.batch_size,
                                             **common), datasets, device)
        else:
            from fizgig.scripts.minimax_cache_text import cache_text
            cache_text(argparse.Namespace(text_encoder=args.model, reference_count=int(aux.get("reference_count", 0)),
                                          no_quantize=aux.get("no_quantize") == "1",
                                          batch_size=args.batch_size or 16, **common), datasets, device)
        return True

    def media_problem(self, path):
        from fizgig.minimax.clip import ClipRejected, is_video, validate
        if not is_video(path):
            return ""
        try:
            validate(path)
        except ClipRejected as e:
            return str(e)
        return ""

    def batch_cond(self, batch, device):
        cond = {"hidden_states": batch["hidden_states"].to(device)}
        if batch.get("audio_latent") is not None:
            cond["audio_latent"] = batch["audio_latent"].to(device)
        if batch.get("audio_only") is not None and bool(batch["audio_only"].any()):
            cond["audio_only"] = True
        return cond

    # ---- training: the old compute_loss ------------------------------------------------------------
    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        from fizgig.minimax.trainer import compute_loss, sample_sigmas
        lat = latents if latents.dim() == 5 else latents.unsqueeze(2)          # (1, 24, T, H, W)
        _pt, ph, pw = getattr(dit, "patch_size", (1, 2, 2))
        tokens = (lat.shape[-2] // ph) * (lat.shape[-1] // pw)
        sigma = sample_sigmas(1, "cpu", generator=generator, image_tokens=tokens)
        if min_t > 0.0 or max_t < 1.0:
            sigma = min_t + (max_t - min_t) * sigma
        noise = torch.randn(lat.shape, generator=generator, dtype=torch.float32)
        audio = cond.get("audio_latent")
        if audio is not None and audio.dim() == 3:
            audio = audio[0]                                                   # cached (2T, 32), batched
        loss, s = compute_loss(dit, lat.to(DTYPE), cond["hidden_states"].to(DTYPE), sigma=sigma.to(lat.device),
                               noise=noise, audio_latent=audio,
                               video_weight=0.0 if cond.get("audio_only") else 1.0)
        return loss, {"t": float(s)}

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        main = [Block(f"h3blk_{i}", f"Block {i}", keep([f"blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                for i in range(self.description.n_blocks)]
        refiner = [Block(f"h3_rf_{i}", f"Token refiner {i}",
                         keep([f"token_refiner.blocks.{i}.{m}" for m in _BLOCK_MODULES])) for i in range(2)]
        return [BlockGroup("Blocks", main), BlockGroup("Token refiner", refiner)]

    def lora_target_names(self, dit):
        """The 50 main blocks (the old default; the token refiner trains only on request)."""
        return [m for b in self.block_map(dit)[0].blocks for m in b.modules]
