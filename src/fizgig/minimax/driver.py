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
TURBO = "h3_turbo"                     # the preview Turbo LoRA's adapter name
_BLOCK_MODULES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


class MiniMaxDriver(FamilyDriver):

    _allowed = {}
    _uncond = None

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

    # ---- the old trainer's training options (--family_option KEY=VALUE) ------------------------------
    #   photo_blocks / clip_blocks / audio_blocks   e.g. 20-49: that modality's steps train only these blocks (the
    #                                               old --photo_blocks / --clip_blocks / --audio_blocks, backward cut)
    #   tread             ratio@start-end, e.g. 0.5@2-47: clip steps route that share of video tokens past the blocks
    #   caption_dropout   e.g. 0.05: swap in the cached empty-prompt embed on that share of steps
    #   shift             the sigma density (the old --shift); unset = H3's own shift 12
    #   clip_still_as_photo  1: each clip's cached still also trains as a photo
    def set_options(self, options):
        super().set_options(options)
        from fizgig.minimax.trainer import parse_block_spec
        n = self.description.n_blocks
        self._allowed = {m: set(parse_block_spec(options[f"{m}_blocks"], n)) for m in ("photo", "clip", "audio")
                         if options.get(f"{m}_blocks")}
        self._uncond = None
        from fizgig.dataset.image_dataset import ImageDataset
        ImageDataset.clip_still_as_photo = options.get("clip_still_as_photo") == "1"

    def prepare_training(self, dit, group, net=None):
        import logging
        import os
        log = logging.getLogger(__name__)
        tpath = self.options.get("turbo_lora")
        if tpath and net is not None:
            # the old previews' Turbo: wired once, OFF and on the CPU between previews; its Linears ride the family
            # LoRA, its full-model AdaLN rows the pruned base's run-time injection (turbo_adaln_patch)
            strength = float(self.options.get("turbo_strength") or 0.75)
            n = net.add_file(tpath, TURBO, strength)
            net.set_enabled(TURBO, False)
            net.move_adapter(TURBO, "cpu")
            self._net, self._turbo_pairs = net, self._adaln_pairs(dit, tpath, strength)
            log.info(f"[turbo] {n} modules at strength {strength:g} + {len(self._turbo_pairs)} adaln via run-time "
                     f"injection; previews {int(self.options.get('turbo_steps') or 6)} steps")
        tread = self.options.get("tread")
        if tread:
            ratio, span = tread.split("@")
            start, end = (int(x) for x in span.split("-"))
            dit._tread = (float(ratio), start, end)
            log.info(f"[tread] token routing ON - {float(ratio) * 100:.0f}% of the video tokens skip blocks {start}-"
                     f"{end - 1} on every clip step")
        p = float(self.options.get("caption_dropout") or 0)
        if p > 0:
            from safetensors.torch import load_file
            for ds in group.datasets:
                f = os.path.join(getattr(ds, "cache_directory", "") or "", "uncond_minimaxh3_te.safetensors")
                if os.path.isfile(f):
                    self._uncond = load_file(f)["hidden_states"].unsqueeze(0)
                    break
            if self._uncond is None:
                log.warning("[caption_dropout] no uncond embed in the cache dirs - dropout disabled for this run")
            else:
                log.info(f"[caption_dropout] {p:.2f} - empty-prompt embed loaded")
        for m, allowed in self._allowed.items():
            log.info(f"[likeness] {m} steps train blocks {min(allowed)}-{max(allowed)} only (backward cut at the "
                     f"window)")

    @staticmethod
    def _modality(batch):
        if batch.get("audio_only") is not None and bool(batch["audio_only"].any()):
            return "audio"
        return "photo" if batch["latents"].dim() == 4 else "clip"

    def step_frozen_blocks(self, batch):
        allowed = self._allowed.get(self._modality(batch))
        if not allowed:
            return ()
        return tuple(f"h3blk_{i}" for i in range(self.description.n_blocks) if i not in allowed)

    def batch_cond(self, batch, device):
        import random
        text = batch["hidden_states"]
        if self._uncond is not None and random.random() < float(self.options.get("caption_dropout") or 0):
            text = self._uncond                                               # caption dropout step
        cond = {"hidden_states": text.to(device)}
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
        shift = self.options.get("shift")
        shift = float(shift) if shift not in (None, "", "sigmoid", "resolution") else (shift or None)
        sigma = sample_sigmas(1, "cpu", shift=shift, generator=generator, image_tokens=tokens)
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

    # ---- previews (first stage: the reference sampler, the old clip contract) -----------------------
    #   options: preview_frames (1 = a still; 22 / 39 / 56 ... clips), preview_audio=1 (a clip's sound),
    #            audio_vae=PATH (the decoder for that sound)
    def load_text_encoder(self, path, device):
        from fizgig.minimax.embedder import load_minimax_h3_te_planned
        return load_minimax_h3_te_planned(path, device=device, compute_dtype=DTYPE, quantize=True)

    def unload_text_encoder(self, te):
        import gc
        del te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"hidden_states": te.encode(c)[0].detach().cpu()} for c in captions]       # (L, 5120)

    def load_vae(self, path, device):
        """The video decoder in fp16 on the CPU (it rides to the GPU for the decode only, as the old previews) and,
        with the audio_vae option, the audio decoder."""
        from safetensors import safe_open
        from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
        dec = MiniMaxH3VideoVAEDecoder()
        with safe_open(path, framework="pt", device="cpu") as f:
            dec.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)
        audio = None
        apath = self.options.get("audio_vae")
        if apath:
            from fizgig.minimax.audio_vae import load_minimax_h3_audio_vae_decoder
            audio = load_minimax_h3_audio_vae_decoder(apath, device="cpu")
        return {"video": dec.to(torch.float16).eval(), "audio": audio, "device": device}

    def initial_noise(self, seed, width, height):
        raise NotImplementedError("H3 travel previews come with the workbench wrap")

    @staticmethod
    def _adaln_pairs(dit, path, strength):
        """A LoRA file's full-model AdaLN rows as (AdalnProj, A, B * strength) - the old _prefilter_frozen_lora's
        AdaLN half (the Linears are the family LoRA's)."""
        from safetensors.torch import load_file
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        sd = ensure_kohya_lora_state_dict(load_file(path))
        parents = {f"lora_unet_{n.replace('.', '_')}_linear": m for n, m in dit.named_modules()
                   if type(m).__name__ == "AdalnProj"}
        out = []
        for name, ap in parents.items():
            down, up = sd.get(f"{name}.lora_down.weight"), sd.get(f"{name}.lora_up.weight")
            lin = ap.linear.base if hasattr(ap.linear, "base") else ap.linear
            if down is not None and up is not None and up.shape[0] == lin.out_features:
                out.append((ap, down.clone(), up.clone() * float(strength)))
        return out

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None, frames=None, audio=None):
        turbo = getattr(self, "_turbo_pairs", None) is not None
        if not turbo:
            return self._generate(dit, cond, width, height, steps, seed, cfg, neg_cond, frames, audio)
        from fizgig.minimax.trainer import turbo_adaln_patch, turbo_adaln_unpatch
        device = next(p for p in dit.parameters() if p.device.type != "meta").device
        try:
            self._net.move_adapter(TURBO, device)
            self._net.set_enabled(TURBO, True)
            turbo_adaln_patch(dit, self._turbo_pairs, device, DTYPE)
            return self._generate(dit, cond, width, height, int(self.options.get("turbo_steps") or 6), seed, 1.0,
                                  None, frames, audio)
        finally:
            turbo_adaln_unpatch(self._turbo_pairs)
            self._net.set_enabled(TURBO, False)
            self._net.move_adapter(TURBO, "cpu")

    def _generate(self, dit, cond, width, height, steps, seed, cfg, neg_cond, frames, audio):
        from fizgig.minimax import sampling
        device = next(p for p in dit.parameters() if p.device.type != "meta").device
        frames = int(self.options.get("preview_frames", 1) if frames is None else frames)
        want_audio = (self.options.get("preview_audio") == "1" if audio is None else bool(audio)) and frames > 1
        txt = cond["hidden_states"]
        txt = (txt[None] if txt.dim() == 2 else txt).to(device, DTYPE)
        unc = None
        if cfg > 1.0 and neg_cond is not None:
            unc = neg_cond["hidden_states"]
            unc = (unc[None] if unc.dim() == 2 else unc).to(device, DTYPE)
        lat, arows = sampling.sample_image(dit, txt, width=width, height=height, steps=steps, cfg_scale=cfg,
                                           uncond_embeds=unc, seed=seed, device=device, dtype=DTYPE,
                                           num_frames=frames, return_audio=True)
        return {"latent": lat.cpu(), "audio": arows.cpu() if (want_audio and arows is not None) else None,
                "size": (width, height)}

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        """A still -> PIL. A clip -> a dict (frames [3, F, H, W] in [0, 1], the middle frame, the waveform) that
        save_preview writes as the old contract."""
        from PIL import Image
        device = vae["device"]
        dec = vae["video"].to(device)
        lat = latents["latent"].to(device).float()
        try:
            if lat.shape[2] > 1:
                px = dec.decode_clip(lat)[0].float().cpu()                    # [3, F, H, W]
                mid = (px[:, px.shape[1] // 2].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
                wave = None
                if latents.get("audio") is not None and vae.get("audio") is not None:
                    from fizgig.minimax.audio_vae import unpack_audio
                    adec = vae["audio"].to(device)
                    wave = adec.decode(unpack_audio(latents["audio"]).to(device, torch.float32))[0].cpu()
                    vae["audio"].to("cpu")
                return {"frames": px, "image": Image.fromarray(mid), "wave": wave}
            px = dec.decode(lat)[0]
            return Image.fromarray((px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
        finally:
            vae["video"].to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_preview(self, result, path):
        """The old previews' contract: a clip writes every 2nd frame as JPEG in <stem>.clip/, the wav and a playable
        mp4 beside it, then the middle frame as the PNG - LAST, the gallery's 'finished' signal."""
        import os
        from PIL import Image
        if not isinstance(result, dict):
            result.save(path)
            return [path]
        stem = path[:-4]
        px, out = result["frames"], []
        clip_dir = stem + ".clip"
        os.makedirs(clip_dir, exist_ok=True)
        keep = list(range(0, px.shape[1], 2))
        if keep[-1] != px.shape[1] - 1:
            keep.append(px.shape[1] - 1)
        for k in keep:
            fr = (px[:, k].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
            Image.fromarray(fr).save(os.path.join(clip_dir, f"f{k:03d}.jpg"), quality=87)
        if result.get("wave") is not None:
            from fizgig.minimax.trainer import write_preview_mp4, write_wav
            write_wav(stem + ".wav", result["wave"])
            out.append(stem + ".wav")
            try:
                write_preview_mp4(stem + ".mp4", px, stem + ".wav")
                out.append(stem + ".mp4")
            except Exception:
                pass                                     # the wav and the scrub frames still work
        result["image"].save(path)
        return out + [path]

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
