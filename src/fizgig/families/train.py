"""Standard-layer LoRA trainer for any described family (fizgig.families), driven through the family's driver.

    python src/fizgig/families/train.py --family qwen_image21 --dit ... --dataset_config ... --output_dir ...

The loop is family-agnostic: Fizgig dataset + bucketing (batch 1), frozen adapters (the family's training adapter,
off for previews; a context LoRA, on for previews; neither in saves), Adaptive LR or a step scheduler, gradient
clipping, optional EMA, per-epoch checkpoints in the family's LoRA key format with SAI metadata, resumable state
dirs and the GUI's pause contract. Everything model-specific (loading, noise/target/timesteps, forward, sampling,
decoding, which Linears a LoRA wraps) comes from the driver.

Why training adapters exist: on Qwen Image 2.1 a plain LoRA collapsed at lr 5e-4 and wobbled at 1e-4; with the same
Adaptive LR a no-adapter run fell to 37 likeness at step 2000 while Fizgig's adapter held 74-77 (26 Sep 2026).
"""
import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from multiprocessing import Value

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,  # noqa: E402
                                   generate_dataset_group_by_blueprint, load_user_config)
from fizgig.families import quant  # noqa: E402
from fizgig.families.lora import FamilyLoRA  # noqa: E402
from fizgig.families.registry import get as get_family  # noqa: E402
from fizgig.krea2.trainer import AdaptiveLR  # noqa: E402
from fizgig.training.metadata import (build_metadata, latest_sample_image, refresh_checkpoint_thumbnail,  # noqa: E402
                                      resolve_title, sample_for_epoch, thumbnail_data_uri)
from fizgig.training.train_utils import LossRecorder, prune_state_dirs  # noqa: E402

logger = logging.getLogger(__name__)

ADAPTER = "training_adapter"
CONTEXT = "context"
SPEED = "speed_lora"


class _Collator:
    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def _step_scheduler(optimizer, kind, warmup, total, cycles=1, power=1.0):
    def f(s):
        if warmup and s < warmup:
            return (s + 1) / warmup
        if kind in ("constant", "constant_with_warmup"):
            return 1.0
        prog = min(1.0, (s - warmup) / max(1, total - warmup))
        if kind == "cosine":
            return 0.5 * (1 + math.cos(math.pi * prog))
        if kind == "cosine_with_restarts":
            return 0.5 * (1 + math.cos(math.pi * ((prog * cycles) % 1.0)))
        if kind == "linear":
            return 1.0 - prog
        if kind == "polynomial":
            return (1.0 - prog) ** power
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _save_state(output_dir, output_name, net, optimizer, *, epoch, global_step, arch_id, extra=None, ema=None):
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    net.save(os.path.join(state_dir, "lora.safetensors"), dtype=torch.float32)
    torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(state_dir, "ema.pt"))
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, os.path.join(state_dir, "rng.pt"))
    with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:   # commit marker, last
        json.dump({"epoch": epoch, "global_step": global_step, "architecture": arch_id, **(extra or {})}, f)
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _load_state(state_dir, net, optimizer, device):
    for need in ("lora.safetensors", "optimizer.pt", "training_state.json"):
        if not os.path.isfile(os.path.join(state_dir, need)):
            raise RuntimeError(f"[resume] {state_dir} is not a saved training state (missing {need}). Pick the "
                               f"folder named like '<lora name>-000012-state'.")
    if net.load_trainable(os.path.join(state_dir, "lora.safetensors")) == 0:
        raise RuntimeError(f"[resume] {state_dir} matched none of this LoRA's modules - different rank or "
                           f"target modules?")
    optimizer.load_state_dict(torch.load(os.path.join(state_dir, "optimizer.pt"), map_location=device))
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        rng = torch.load(rng_path)
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    with open(os.path.join(state_dir, "training_state.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0)), meta


def _preview_vram(tag, reset_peak=False):
    """One line of VRAM state at a preview waypoint (the same lines Krea 2 logs, #123)."""
    try:
        if not torch.cuda.is_available():
            return
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        a = torch.cuda.memory_allocated() / 1024 ** 3
        r = torch.cuda.memory_reserved() / 1024 ** 3
        pk = torch.cuda.max_memory_reserved() / 1024 ** 3
        f = torch.cuda.mem_get_info()[0] / 1024 ** 3
        logger.info(f"[preview-vram] {tag}: allocated {a:.2f} GB, reserved {r:.2f} GB (peak {pk:.2f} GB), "
                    f"free {f:.2f} GB")
    except Exception:
        pass


def _small_card_previews():
    """Cards under 20 GB get the low-memory preview treatment (Krea 2's rule): the training DiT parks on CPU for
    the VAE decode and the preview canvas caps at 768 px. FIZGIG_PREVIEW_LOWMEM=1/0 forces it; FIZGIG_SIM_VRAM_GB
    simulates a smaller card."""
    ov = os.environ.get("FIZGIG_PREVIEW_LOWMEM", "").strip()
    if ov in ("0", "1"):
        return ov == "1"
    try:
        if not torch.cuda.is_available():
            return False
        sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
        total = float(sim) if sim else torch.cuda.get_device_properties(0).total_memory / 1e9
        return total < 20.0
    except Exception:
        return False


def _read_sample_override(output_dir):
    """The GUI's live sample override (<output_dir>/.sample_override.json, written by the status-bar panel):
    {prompt, seed, width, height} while a prompt is set, else None. The reference image field is Klein's and ignored."""
    path = os.path.join(output_dir, ".sample_override.json")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        if prompt:
            return {"prompt": prompt, "seed": int(d.get("seed", 1234)), "width": int(d.get("width", 1024)),
                    "height": int(d.get("height", 1024))}
    except (OSError, ValueError, TypeError):
        pass
    return None


def _encode_override(driver, te_path, prompt, dit, device, parkable=True, references=None):
    """Encode one override prompt mid-run. The text encoder loads beside the training DiT when it fits; otherwise the
    DiT waits on CPU for the encode (a block-swapped DiT is small enough to stay, and the driver picks a smaller
    encoder when VRAM is short)."""
    from fizgig.families import quant
    te_gb = os.path.getsize(te_path) / 1024 ** 3 if te_path and os.path.exists(te_path) else 0.0
    park = parkable and quant.free_vram_gb() < te_gb + 2.0
    if park:
        quant.move(dit, "cpu")
        torch.cuda.empty_cache()
    try:
        if references:
            te = driver.load_reference_text_encoder(te_path, device)
        else:
            te = driver.load_text_encoder(te_path, device)
        try:
            if references:
                return driver.encode_text_with_references(te, [prompt], [references])
            return driver.encode_text(te, [prompt])
        finally:
            driver.unload_text_encoder(te)
            del te
            torch.cuda.empty_cache()
    finally:
        if park:
            quant.move(dit, device)


def _cap_canvas(width, height, cap=768):
    long = max(width, height)
    if long <= cap:
        return width, height
    s = cap / long
    return max(64, int(width * s) // 32 * 32), max(64, int(height * s) // 32 * 32)


@torch.no_grad()
def _reference_size(w, h, area):
    """An edit preview's canvas: the reference's aspect at `area` pixels, in multiples of 64 (a training bucket's
    step, so the reference's vision tokens and latents line up as they do in training)."""
    r = w / h
    return max(64, round((area * r) ** 0.5 / 64) * 64), max(64, round((area / r) ** 0.5 / 64) * 64)


def _load_references(paths, width, height):
    """uint8 (H, W, 3) arrays of the preview's reference images at the first one's aspect and the preview's area,
    and that (width, height)."""
    import numpy as np
    from PIL import Image, ImageOps
    imgs = [Image.open(p).convert("RGB") for p in paths]
    size = _reference_size(*imgs[0].size, width * height)
    return [np.array(ImageOps.fit(im, size, Image.LANCZOS)) for im in imgs], size


class _SliderBank(torch.utils.data.Dataset):
    """Prompt-pair sliders train on no dataset: the base model renders a bank of practice latents from the neutral
    prompt, and each step noises one of them."""

    def __init__(self, latents):
        self.latents = latents
        self.datasets = []
        self.num_train_items = len(latents)

    def set_current_epoch(self, epoch):
        pass

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        return {"latents": self.latents[i]}


# The dial is the point, so a slider preview shows it moving: one prompt, one seed, these strengths side by side.
SLIDER_PREVIEW_MULTIPLIERS = (-1.0, 0.0, 1.0)


def _slider_strip(frames, multipliers):
    from PIL import Image, ImageDraw, ImageFont
    w, h = frames[0].size
    gap, band = 8, 30
    strip = Image.new("RGB", (w * len(frames) + gap * (len(frames) - 1), h + band), (16, 16, 16))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default(size=max(14, w // 40))
    except Exception:
        font = ImageFont.load_default()
    for k, (im, m) in enumerate(zip(frames, multipliers)):
        x = k * (w + gap)
        strip.paste(im, (x, band))
        draw.text((x + 8, 7), "strength 0 (slider off)" if m == 0 else f"strength {m:+g}", fill=(236, 236, 236),
                  font=font)
    return strip


def _pair_slider_caption(user_config):
    """The caption an image-pair slider's photos share (the most common one if they were edited apart)."""
    from collections import Counter
    general = user_config.get("general", {})
    counts = Counter()
    for d in user_config.get("datasets", []):
        folder = d.get("image_directory")
        ext = d.get("caption_extension") or general.get("caption_extension") or ".txt"
        if not folder or not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(ext):
                with open(os.path.join(folder, f), encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
                if text:
                    counts[text] += 1
    return counts.most_common(1)[0][0] if counts else None


def _prompt_slider_step(driver, dit, net, latents, enc, gen, *, guidance, min_t, max_t):
    """Concept Sliders, textual form, on flow matching. With the adapter at 0 the frozen model predicts the
    neutral, positive and negative prompts at one noised practice latent; the adapter then trains at +1 toward
    v_neutral + guidance * (v_pos - v_neg) and at -1 toward the mirror. Each pole is backpropagated before the
    flip (checkpointed blocks recompute at the strength of the moment). Returns the detached mean loss."""
    state = driver.noise_latents(latents, gen, min_t=min_t, max_t=max_t)
    n_c, p_c, g_c = enc
    try:
        net.set_trainable_multiplier(0.0)
        with torch.no_grad():
            v_n = driver.predict(dit, state, n_c).float()
            delta = float(guidance) * (driver.predict(dit, state, p_c).float() - driver.predict(dit, state, g_c).float())
        total = 0.0
        for m in (1.0, -1.0):
            net.set_trainable_multiplier(m)
            loss = F.mse_loss(driver.predict(dit, state, n_c).float(), v_n + m * delta)
            (0.5 * loss).backward()
            total += 0.5 * loss.item()
    finally:
        net.set_trainable_multiplier(1.0)
    return total, state["t"]


def _render_previews(driver, dit, net, vae, encoded, out_dir, epoch, *, output_name, steps, cfg, neg, width,
                     height, seed, ema=None, speed=None, lowmem=False, swapped=False, refs=None, slider=False):
    """Previews on the RESIDENT training model: training adapter OFF (the deployment setup), the family's speed
    LoRA ON if one is loaded (it lives on CPU between previews). On small cards the DiT parks on CPU for the
    decode. File names match the other trainers so the GUI gallery reads them:
    <name>_e<epoch>_<idx>_<timestamp>_<seed>.png. `speed` is the speed LoRA's SamplingSettings or None; `refs` the edit previews' reference latents; `slider`
    renders each prompt at -1 / 0 / +1 on one seed and saves them as one strip."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device = next(iter(dit.parameters())).device
    _preview_vram("preview start", reset_peak=True)
    net.set_enabled(ADAPTER, False)
    if speed is not None:
        net.move_adapter(SPEED, device)
        net.set_enabled(SPEED, True)
    if ema is not None:
        ema.swap_in()
    was_training = dit.training
    dit.eval()
    if swapped:
        driver.block_swap_mode(dit, inference=True)
    paths = []
    ref_kw = {"refs": [r.to(device) for r in refs]} if refs else {}
    try:
        lats = []
        mults = SLIDER_PREVIEW_MULTIPLIERS if slider else (None,)
        for i, cond in enumerate(encoded):
            for m in mults:
                if m is not None:
                    net.set_trainable_multiplier(m)
                if speed is not None:
                    lats.append(driver.generate(dit, cond, width, height, steps=steps, seed=seed + i, cfg=speed.cfg,
                                                sigmas=speed.sigmas, options=speed.options, **ref_kw))
                else:
                    lats.append(driver.generate(dit, cond, width, height, steps=steps, seed=seed + i, cfg=cfg,
                                                neg_cond=neg, **ref_kw))
        if slider:
            net.set_trainable_multiplier(1.0)
        park = lowmem and not swapped      # a swapped DiT is already mostly on CPU; moving it would undo the layout
        if park:                            # #123: never hold the training DiT and the VAE decode together
            _preview_vram("before decode")
            quant.move(dit, "cpu")
            torch.cuda.empty_cache()
            _preview_vram("DiT parked for the decode")
        per = len(SLIDER_PREVIEW_MULTIPLIERS) if slider else 1
        for i in range(len(lats) // per):
            p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
            frames = [driver.decode(vae, lat, width, height) for lat in lats[i * per:(i + 1) * per]]
            (_slider_strip(frames, SLIDER_PREVIEW_MULTIPLIERS) if slider else frames[0]).save(p)
            paths.append(p)
    finally:
        if lowmem and not swapped:
            quant.move(dit, device)
            _preview_vram("after decode, DiT restored")
        if swapped:
            driver.block_swap_mode(dit, inference=False)
        if ema is not None:
            ema.swap_out()
        if speed is not None:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
        net.set_enabled(ADAPTER, True)
        dit.train(was_training)
        torch.cuda.empty_cache()
        _preview_vram("after preview cleanup")
    logger.info(f"[sample] epoch {epoch}: {len(paths)} preview(s) -> {out_dir}")
    return paths


def train_family(family, dit_path, dataset_config, output_dir, output_name, *, network_dim=32, network_alpha=32,
                 learning_rate=1e-4, max_train_epochs=16, save_every_n_epochs=1, save_state=False,
                 save_state_on_train_end=False, keep_last_n_states=2, seed=42, precision="bf16",
                 training_adapter=None, training_adapter_strength=1.0,
                 context_lora_path=None, context_lora_strength=1.0, min_timestep=0.0, max_timestep=1.0,
                 speed_lora=None, speed_lora_strength=None,
                 vae_path=None, te_path=None, sample_prompts=None, sample_every_n_epochs=0, sample_width=None,
                 sample_height=None, sample_steps=None, sample_cfg_scale=None, sample_negative=None,
                 sample_at_first=False, sample_seed=42, sample_reference=None,
                 slider_pairs=False, slider_diff_weight=1.0, slider_prompts=None, slider_guidance=3.0,
                 slider_bank=16, slider_bank_res=768,
                 metadata_title=None, metadata_author=None, metadata_description=None, metadata_license=None,
                 metadata_tags=None, metadata_trigger_phrase=None, metadata_thumbnail=None,
                 resume_state_dir=None, adaptive_lr=False, adaptive_lr_min=1e-4, adaptive_lr_max=2e-4,
                 max_grad_norm=1.0, ema_decay=0.0, optimizer_type="adamw", optimizer_args="",
                 lr_scheduler="constant", lr_warmup_steps=0, lr_scheduler_num_cycles=1, lr_scheduler_power=1.0,
                 gradient_checkpointing=True, blocks_to_swap=0, network_type="lora", lokr_factor=8,
                 log_per_image_loss=False, per_image_lr=False, auto_recaption=False, warmup_look_outliers=False,
                 trigger_word=None, trigger_position="start", recaption_instruction=None,
                 recaption_instruction_detailed=None, captioner=None):
    desc = get_family(family)
    if desc is None or not desc.training_ready:
        raise RuntimeError(f"unknown or untrainable family {family!r}")
    driver = desc.load_driver()
    arch = desc.arch_id
    speed_desc = desc.preview_speed() if speed_lora else None
    if speed_lora and speed_desc is None:
        logger.warning(f"[sample] {desc.display_name} declares no preview speed LoRA - ignoring --speed_lora")
        speed_lora = None
    if speed_desc is not None and speed_lora_strength is not None and speed_lora_strength <= 0:
        logger.info(f"[sample] {speed_desc.name} at strength 0 - previews render without it")
        speed_lora = speed_desc = None
    # --speed_lora on its own means the turbo's own recipe (strength, steps); the GUI passes its Samples-tab values
    sample_steps = sample_steps or (speed_desc.settings.steps if speed_desc else desc.preview_steps)
    sample_cfg_scale = desc.preview_cfg if sample_cfg_scale is None else sample_cfg_scale
    sample_width = sample_width or desc.preview_width
    sample_height = sample_height or desc.preview_height
    lowmem = _small_card_previews()
    if lowmem and max(sample_width, sample_height) > 768:
        sample_width, sample_height = _cap_canvas(sample_width, sample_height)
        logger.info(f"[preview] card under 20 GB: preview canvas capped to {sample_width}x{sample_height} and the "
                    f"DiT parks on CPU for the decode (#123). FIZGIG_PREVIEW_LOWMEM=0 turns this off.")
    device = torch.device("cuda")
    quant.apply_vram_cap()          # FIZGIG_SIM_VRAM_GB: behave like a smaller card
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ---- sliders: a signed dial, trained at +1 and -1 ------------------------------------------------
    slider = bool(slider_pairs or slider_prompts)
    if slider:
        if not desc.slider_training:
            raise RuntimeError(f"{desc.display_name} does not offer slider training")
        if slider_pairs and slider_prompts:
            raise RuntimeError("[slider] image pairs and prompt pairs are two different sliders - pick one")
        if slider_prompts and (len(slider_prompts) != 3 or not all(str(x).strip() for x in slider_prompts[1:])):
            raise RuntimeError("[slider] prompt pairs need NEUTRAL POSITIVE NEGATIVE (the two poles non-empty)")
        if network_type != "lora":
            logger.info("[slider] network type %s -> LoRA: a slider is a plain LoRA whose strength is the dial",
                        network_type)
            network_type = "lora"
        if str(optimizer_type).lower().startswith("automagic"):
            logger.info("[slider] optimizer %s -> adamw8bit: the +1/-1 flip every step reads as noise to an "
                        "optimizer that sets its own rate", optimizer_type)
            optimizer_type = "adamw8bit"
        if adaptive_lr or ema_decay or log_per_image_loss or per_image_lr or auto_recaption or warmup_look_outliers:
            logger.info("[slider] adaptive LR, weight averaging and the per-image loss watch are off: they assume "
                        "one target per image, and a slider step has two")
        adaptive_lr, ema_decay = False, 0.0
        log_per_image_loss = per_image_lr = auto_recaption = warmup_look_outliers = False
        if slider_pairs:
            logger.info("[slider] IMAGE-PAIR SLIDER: the adapter trains at +1 toward each image and at -1 toward "
                        "its pair (difference weight %g)", float(slider_diff_weight))
        else:
            logger.info("[slider] PROMPT-PAIR SLIDER: base '%s' | +1 -> '%s' | -1 -> '%s' (guidance %g)",
                        *slider_prompts, float(slider_guidance))
            if not (te_path and vae_path):
                raise RuntimeError("[slider] prompt pairs need the text encoder and VAE paths")
            if str(slider_prompts[0]).strip():     # the dial is shown on the picture it was trained on
                sample_prompts = [str(slider_prompts[0]).strip()]

    # ---- data ------------------------------------------------------------------------------------
    shared_epoch = Value("i", 0)
    if slider_prompts:
        user_config = {"general": {"resolution": [slider_bank_res, slider_bank_res]}}
        group = _SliderBank([None] * max(1, int(slider_bank)))    # filled once the DiT is loaded
    else:
        if not dataset_config:
            raise RuntimeError("--dataset_config is required (only a prompt-pair slider trains without a dataset)")
        user_config = load_user_config(dataset_config)
        blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, argparse.Namespace(),
                                                                   architecture=arch)
        group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True,
                                                    num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items - run the cache stages (families/cache.py) first.")
    if slider_pairs:
        shared = _pair_slider_caption(user_config)
        if shared:                           # the dial is shown on what both ends of a pair have in common
            sample_prompts = [shared]
            logger.info("[slider] previews use the pairs' caption: '%s'", shared)
    for ds in group.datasets:
        if getattr(ds, "batch_size", 1) != 1:
            raise RuntimeError(f"{desc.display_name} trains at batch size 1 here (conditioning lengths differ per "
                               f"image). Set Batch Size to 1.")
    loader = DataLoader(group, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=(lambda b: b[0]) if slider_prompts else _Collator(shared_epoch, group))
    steps_per_epoch = len(loader)
    logger.info(f"{desc.display_name} training: {group.num_train_items} items, {max_train_epochs} epochs, "
                f"{steps_per_epoch} steps/epoch")

    # ---- Auto precision / swap: planned on an empty card (the description's figures include the preview VAE)
    if precision == "auto" or blocks_to_swap < 0:
        req = (precision, blocks_to_swap)
        res = user_config.get("general", {}).get("resolution") or [1024, 1024]
        mp = (res[0] * res[1] if isinstance(res, (list, tuple)) else res * res) / 1e6
        precision, blocks_to_swap, why = quant.plan(desc, driver, precision, blocks_to_swap, megapixels=mp)
        why += f" at {mp:.2f} MP"
        logger.info(f"[precision] Auto plan: {precision}, block swap {blocks_to_swap} ({why}); asked {req}")

    # ---- previews: encode prompts once, keep the VAE ---------------------------------------------
    encoded = neg = vae = ref_imgs = ref_latents = None
    slider_enc = None
    if slider_prompts:
        te = driver.load_text_encoder(te_path, device)
        slider_enc = [{k: v[None] for k, v in c.items()}
                      for c in driver.encode_text(te, [str(x) for x in slider_prompts])]
        driver.unload_text_encoder(te)
        del te
        torch.cuda.empty_cache()
    sample_dir = os.path.join(output_dir, "sample")
    if sample_reference and not driver.supports_references:
        logger.warning(f"[sample] {desc.display_name} has no edit previews - ignoring --sample_reference")
        sample_reference = None
    if sample_prompts and sample_every_n_epochs and te_path and vae_path:
        logger.info("[sample] encoding %d preview prompt(s) with %s", len(sample_prompts), desc.text_encoder_label)
        if sample_reference:
            ref_imgs, (sample_width, sample_height) = _load_references(sample_reference, sample_width, sample_height)
            logger.info(f"[sample] edit previews from {len(ref_imgs)} reference image(s) at "
                        f"{sample_width}x{sample_height}")
            te = driver.load_reference_text_encoder(te_path, device)
            encoded = driver.encode_text_with_references(te, sample_prompts, [ref_imgs] * len(sample_prompts))
            if sample_cfg_scale > 1.0:
                neg = driver.encode_text_with_references(te, [sample_negative or ""], [ref_imgs])[0]
        else:
            te = driver.load_text_encoder(te_path, device)
            encoded = driver.encode_text(te, sample_prompts)
            if sample_cfg_scale > 1.0:
                neg = driver.encode_text(te, [sample_negative or ""])[0]
        driver.unload_text_encoder(te)
        del te
        torch.cuda.empty_cache()
        vae = driver.load_vae(vae_path, device)
        if ref_imgs:
            ref_latents = [z[None].cpu() for z in driver.encode_images(vae, ref_imgs)]
    elif sample_prompts and sample_every_n_epochs:
        logger.warning("[sample] previews need the text encoder and VAE paths - previews are off for this run")

    # ---- model ----------------------------------------------------------------------------------
    logger.info(f"Loading {desc.display_name} DiT ({precision}) from {dit_path}")
    dit, swapped = quant.load_base(driver, dit_path, device, precision, blocks_to_swap)
    if gradient_checkpointing:
        driver.enable_gradient_checkpointing(dit, True)
    net = FamilyLoRA(dit, driver, device=device)
    if training_adapter:
        n = net.add_file(training_adapter, ADAPTER, training_adapter_strength)
        if n == 0:
            raise RuntimeError(f"Training adapter {training_adapter} matched no {desc.display_name} modules.")
        logger.info(f"[adapter] training adapter ON ({n} Linears, strength {training_adapter_strength:g}) - frozen, "
                    f"off in previews, not saved into the LoRA")
    else:
        logger.warning("[adapter] no training adapter for this run")
    if context_lora_path:
        n = net.add_file(context_lora_path, CONTEXT, context_lora_strength)
        logger.info(f"[context] {os.path.basename(context_lora_path)} frozen + active at {context_lora_strength:g} "
                    f"({n} Linears)")
    if speed_lora and encoded is not None:
        n = net.add_file(speed_lora, SPEED, speed_desc.strength if speed_lora_strength is None else speed_lora_strength)
    if speed_lora and encoded is not None and n == 0:
        net.remove(SPEED)
        logger.warning(f"[sample] {os.path.basename(speed_lora)} matched no {desc.display_name} layers "
                       f"(a turbo LoRA for another model?) - previews render without it, at "
                       f"{desc.preview_steps} steps")
        speed_lora = None
        sample_steps = desc.preview_steps
    elif speed_lora and encoded is not None:
        net.set_enabled(SPEED, False)
        net.move_adapter(SPEED, "cpu")
        logger.info(f"[sample] {speed_desc.name}: {n} Linears, on CPU between previews, on only while they render "
                    f"({sample_steps} steps, strength {speed_desc.strength if speed_lora_strength is None else speed_lora_strength:g})")
    if network_type == "lokr" and "lokr" not in desc.network_types:
        raise RuntimeError(f"{desc.display_name} does not offer LoKR")
    net.add_trainable(network_dim, network_alpha, kind=network_type, factor=lokr_factor)
    if slider_prompts:
        # the practice bank: the base model's own renders of the neutral prompt (adapter at 0, training adapter
        # off, the speed LoRA on when it is loaded), decoded and re-encoded into training latents
        import numpy as np
        if vae is None:
            vae = driver.load_vae(vae_path, device)
        n_c = {k: v[0].to(device) for k, v in slider_enc[0].items()}
        net.set_trainable_multiplier(0.0)
        net.set_enabled(ADAPTER, False)
        use_speed = speed_lora is not None and speed_desc is not None and net.has(SPEED)
        if use_speed:
            net.move_adapter(SPEED, device)
            net.set_enabled(SPEED, True)
        dit.eval()
        bank = []
        with torch.no_grad():
            for i in tqdm(range(len(group)), desc="[slider] practice images"):
                if use_speed:
                    lat = driver.generate(dit, n_c, slider_bank_res, slider_bank_res, steps=speed_desc.settings.steps,
                                          seed=seed + 1000 + i, cfg=speed_desc.settings.cfg,
                                          sigmas=speed_desc.settings.sigmas, options=speed_desc.settings.options)
                else:
                    lat = driver.generate(dit, n_c, slider_bank_res, slider_bank_res, steps=desc.preview_steps,
                                          seed=seed + 1000 + i, cfg=desc.preview_cfg)
                img = driver.decode(vae, lat, slider_bank_res, slider_bank_res)
                bank.append(driver.encode_images(vae, [np.array(img)])[0][None].cpu())
        if use_speed:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
        net.set_enabled(ADAPTER, True)
        net.set_trainable_multiplier(1.0)
        group.latents = bank
        torch.cuda.empty_cache()
        logger.info("[slider] %d practice images rendered at %dx%d", len(bank), slider_bank_res, slider_bank_res)
    params = net.parameters()
    logger.info((f"LoKR factor {lokr_factor}" if network_type == "lokr" else
                 f"LoRA rank {network_dim} alpha {network_alpha:g}") +
                f": {len(net.trainable_modules())} modules, {sum(p.numel() for p in params) / 1e6:.1f}M trainable params")

    from fizgig.training.optimizers import create_optimizer, owns_its_rate
    optimizer, opt_label = create_optimizer(optimizer_type, params, learning_rate, optimizer_args)
    if owns_its_rate(optimizer):        # Automagic v3 sets its own rate: the watcher and schedulers stand down
        if adaptive_lr:
            logger.info("[adaptive_lr] ignored - the optimizer sets its own learning rate")
        if per_image_lr or warmup_look_outliers:
            logger.info("[per-image LR] per-image LR and the look warm-up are off - the optimizer sets its own rate")
        adaptive_lr = per_image_lr = warmup_look_outliers = False
        logger.info(f"[optimizer] {opt_label} owns the learning rate from here ({learning_rate:.2e} is its start); "
                    f"the LR scheduler stands down")
    if adaptive_lr:                     # the watcher owns the rate: start at the geometric midpoint of Min/Max
        learning_rate = math.sqrt(adaptive_lr_min * adaptive_lr_max)
        for g in optimizer.param_groups:
            g["lr"] = learning_rate
        logger.info(f"[adaptive_lr] ENABLED - start_lr={learning_rate:.3e} min_lr={adaptive_lr_min:.3e} "
                    f"max_lr={adaptive_lr_max:.3e} (the Learning Rate box is ignored)")
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    ema = None
    if ema_decay and ema_decay > 0:
        from fizgig.training.ema import EMAWeights
        ema = EMAWeights(net, float(ema_decay))
        logger.info(f"[ema] ON at decay {ema_decay:g} - checkpoints and previews use the running average")

    start_epoch = global_step = 0
    if resume_state_dir:
        start_epoch, global_step, meta = _load_state(resume_state_dir, net, optimizer, device)
        if adaptive:
            adaptive.load_state_dict(meta.get("adaptive_lr_state"))
        if ema is not None and os.path.exists(os.path.join(resume_state_dir, "ema.pt")):
            ema.load_state_dict(torch.load(os.path.join(resume_state_dir, "ema.pt"), map_location="cpu"))
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch {start_epoch + 1}/{max_train_epochs}")
    from fizgig.families.loss_watch import Watch
    watch = Watch(output_dir, group, user_config, driver, log=log_per_image_loss, per_image_lr=per_image_lr,
                  auto_recaption=auto_recaption, warmup_look=warmup_look_outliers,
                  resume=bool(resume_state_dir), start_epoch=start_epoch, te_path=te_path,
                  trigger_word=trigger_word, trigger_position=trigger_position,
                  recaption_instruction=recaption_instruction,
                  recaption_instruction_detailed=recaption_instruction_detailed, captioner_path=captioner)
    scheduler = None
    if not adaptive and not owns_its_rate(optimizer):
        scheduler = _step_scheduler(optimizer, lr_scheduler, lr_warmup_steps, steps_per_epoch * max_train_epochs,
                                    lr_scheduler_num_cycles, lr_scheduler_power)
        import warnings
        with warnings.catch_warnings():
            # a resume moves the schedule to where the run paused before the first optimizer step, on purpose;
            # PyTorch's "lr_scheduler.step() before optimizer.step()" warning is for training loops, not this
            warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")
            for _ in range(global_step):
                scheduler.step()

    last_prompt = [None]

    def metadata(epoch):
        thumb = None if (metadata_thumbnail or "").lower() in ("off", "none") else (
            metadata_thumbnail or latest_sample_image(output_dir))
        md = build_metadata(None, arch, time.time(),
                            title=metadata_title if metadata_title is not None else resolve_title(
                                output_name, metadata_trigger_phrase),
                            author=metadata_author,
                            description=metadata_description if metadata_description is not None else last_prompt[0],
                            license=metadata_license, tags=metadata_tags, trigger_phrase=metadata_trigger_phrase,
                            thumbnail=thumbnail_data_uri(thumb))
        md.update({"ss_network_module": f"fizgig.families ({desc.key}, {network_type})",
                   "ss_network_dim": str(network_dim if network_type == "lora" else lokr_factor),
                   "ss_network_alpha": str(network_alpha if network_type == "lora" else 1.0),
                   **({"ss_lokr_factor": str(lokr_factor)} if network_type == "lokr" else {}),
                   "ss_architecture": arch, "ss_epoch": str(epoch),
                   "ss_optimizer": opt_label, "ss_learning_rate": f"{learning_rate:g}",
                   "ss_training_adapter": os.path.basename(training_adapter) if training_adapter else "none"})
        if context_lora_path:
            md.update({"ss_context_lora": os.path.basename(context_lora_path),
                       "ss_context_lora_strength": str(context_lora_strength)})
        if slider:
            # the deploy contract: strength is the dial (-1 one pole, +1 the other); tools read this
            md.update({"ss_slider": "prompt_pairs" if slider_prompts else "image_pairs"})
            if slider_prompts:
                md.update({"ss_slider_prompts": json.dumps([str(x) for x in slider_prompts]),
                           "ss_slider_guidance": f"{float(slider_guidance):g}"})
            else:
                md.update({"ss_slider_diff_weight": f"{float(slider_diff_weight):g}"})
        return md

    def save_lora(path, epoch):
        if ema is not None:
            ema.swap_in()
        try:
            net.save(path, metadata(epoch))
        finally:
            if ema is not None:
                ema.swap_out()
        logger.info(f"[save] {path}")

    def previews(epoch):
        if encoded is None:
            return
        conds, w, h, sd, prompts = encoded, sample_width, sample_height, sample_seed, sample_prompts
        ov = _read_sample_override(output_dir)
        if ov:
            logger.info(f"[sample override] active - '{ov['prompt'][:60]}' seed={ov['seed']} {ov['width']}x{ov['height']}")
            try:
                conds = _encode_override(driver, te_path, ov["prompt"], dit, device, parkable=not swapped,
                                         references=ref_imgs)
                sd, prompts = ov["seed"], [ov["prompt"]]
                if not ref_imgs:        # an edit keeps the reference's canvas: its latents are already encoded
                    w, h = ov["width"], ov["height"]
                    if lowmem and max(w, h) > 768:
                        w, h = _cap_canvas(w, h)
            except Exception:
                logger.exception("[sample override] could not encode the override prompt - using the configured ones")
                conds = encoded
        if not sd:          # seed 0 = a fresh random seed every preview round, as on the other trainers
            sd = random.randint(1, 2 ** 31 - 1)
        _render_previews(driver, dit, net, vae, conds, sample_dir, epoch, output_name=output_name,
                         steps=sample_steps, cfg=sample_cfg_scale, neg=neg, width=w, height=h,
                         seed=sd, ema=ema,
                         speed=speed_desc.settings if (speed_lora and speed_desc) else None, lowmem=lowmem,
                         swapped=bool(swapped), refs=ref_latents, slider=slider)
        last_prompt[0] = prompts[-1] if prompts else None

    def state(epoch):
        _save_state(output_dir, output_name, net, optimizer, epoch=epoch, global_step=global_step, arch_id=arch,
                    ema=ema, extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)

    if sample_at_first and start_epoch == 0:
        previews(0)

    # ---- train ----------------------------------------------------------------------------------
    gen = torch.Generator().manual_seed(seed + start_epoch)
    pause_flag = os.path.join(output_dir, ".pause_requested")
    recorder = LossRecorder()
    progress = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step, desc="steps", smoothing=0)
    dit.train()
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i, batch in enumerate(loader):
            if watch.excluded(batch):          # two failed AI recaptions and still stuck: no forward, no loss
                recorder.drop(step=i)
                global_step += 1
                progress.update(1)
                continue
            latents = batch["latents"].to(device)
            if slider:
                # both poles backpropagate before the strength flips: checkpointed blocks recompute their forward
                # in backward, at whatever strength is set then
                optimizer.zero_grad(set_to_none=True)
                if slider_prompts:
                    _l, _t = _prompt_slider_step(driver, dit, net, latents, slider_enc, gen,
                                                 guidance=slider_guidance, min_t=min_timestep, max_t=max_timestep)
                else:
                    cond = {k[len("cond__"):]: v.to(device) for k, v in batch.items() if k.startswith("cond__")}
                    _neg = batch.get("latents_control_0")
                    if _neg is None:
                        raise RuntimeError("[slider] this item has no pair image - an image-pair slider needs one in "
                                           "the second folder for every training image, matched by file name")
                    _neg = _neg.to(device)
                    _l = 0.0
                    try:
                        for m, a, b in ((1.0, latents, _neg), (-1.0, _neg, latents)):
                            net.set_trainable_multiplier(m)
                            _pl, _info = driver.training_loss(dit, a, cond, gen, min_t=min_timestep,
                                                              max_t=max_timestep, diff_ref=b,
                                                              diff_weight=slider_diff_weight)
                            (0.5 * _pl).backward()
                            _l += 0.5 * _pl.item()
                    finally:
                        net.set_trainable_multiplier(1.0)
                    _t = _info.get("t", 0.5)
                loss, _info = torch.tensor(_l), {"t": _t}
            else:
                cond = {k[len("cond__"):]: v.to(device) for k, v in batch.items() if k.startswith("cond__")}
                refs = [batch[k].to(device) for k in sorted((k for k in batch if k.startswith("latents_control_")),
                                                            key=lambda k: int(k.rsplit("_", 1)[1]))]
                loss, _info = driver.training_loss(dit, latents, cond, gen, min_t=min_timestep, max_t=max_timestep,
                                                   **({"refs": refs} if refs else {}))
                optimizer.zero_grad(set_to_none=True)
                mult = watch.multiplier(batch)     # per-image LR (batch size 1): the raw loss is still what's recorded
                (loss * mult if mult != 1.0 else loss).backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update()
            global_step += 1
            recorder.add(epoch=epoch, step=i, loss=loss.item())
            watch.observe(epoch + 1, global_step, batch, _info.get("t", 0.5), loss.item())
            progress.set_postfix(avr_loss=f"{recorder.moving_average:.4f}", refresh=False)
            progress.update(1)

        logger.info(f"epoch {epoch + 1}/{max_train_epochs}  avr_loss={recorder.moving_average:.4f}  step={global_step}  "
                    f"{(time.time() - t0) / max(1, steps_per_epoch):.2f}s/step  "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}  "
                    f"peak VRAM {torch.cuda.max_memory_reserved() / 1024 ** 3:.1f} GB")
        if adaptive:
            adaptive.epoch_boundary(epoch, recorder.moving_average, net.trainable_modules(), optimizer)
        # problem-image verdicts + queued caption fixes / auto-recaptions, re-encoded before the next epoch
        watch.boundary(epoch + 1, dit, device, parkable=not swapped)

        done = epoch + 1
        cadence = bool(save_every_n_epochs) and done % save_every_n_epochs == 0 and done < max_train_epochs
        if cadence:
            save_lora(os.path.join(output_dir, f"{output_name}-{done:06d}.safetensors"), done)
        state_saved = False
        if save_state and cadence:
            state(done)
            prune_state_dirs(output_dir, output_name, keep_last_n_states)
            state_saved = True
        if sample_every_n_epochs and done % sample_every_n_epochs == 0:
            _tp = time.time()
            previews(done)
            progress.start_t += time.time() - _tp       # the bar's s/it is training speed, not previews
            # this epoch's checkpoint was saved before its preview existed, with the previous epoch's as its
            # thumbnail (#122): re-embed its own. An explicit --metadata_thumbnail (or "off") stays.
            if cadence and not (metadata_thumbnail or "").strip():
                own = sample_for_epoch(output_dir, output_name, done)
                if own:
                    refresh_checkpoint_thumbnail(os.path.join(output_dir, f"{output_name}-{done:06d}.safetensors"), own)
        if os.path.exists(pause_flag) and done < max_train_epochs:
            if state_saved:
                logger.info(f"[pause] requested - state for epoch {done} already saved; exiting cleanly")
            else:
                logger.info(f"[pause] requested - saving state at epoch {done} and exiting cleanly")
                state(done)
            progress.close()
            sys.exit(0)

    progress.close()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    save_lora(final, max_train_epochs)
    if save_state_on_train_end:
        state(max_train_epochs)
    logger.info(f"Training complete -> {final}")
    return final


def setup_parser():
    p = argparse.ArgumentParser(description="LoRA training for a described model family (standard layer)")
    p.add_argument("--family", required=True, help="family key, e.g. qwen_image21")
    p.add_argument("--dit", required=True)
    p.add_argument("--dataset_config", default=None, help="The dataset TOML (not used by a prompt-pair slider)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--precision", default="bf16", choices=("auto",) + quant.PRECISIONS,
                   help="base precision: bf16, int8 (8-bit, int8 matmuls), nf4 (4-bit) or auto (fits free VRAM)")
    p.add_argument("--blocks_to_swap", type=int, default=0,
                   help="blocks streamed between CPU and GPU (not with nf4); -1 = as few as fit free VRAM")
    p.add_argument("--network_dim", type=int, default=32)
    p.add_argument("--network_type", default="lora", choices=("lora", "lokr"))
    p.add_argument("--speed_lora_strength", type=float, default=None,
                   help="the preview speed LoRA's strength (default: the family's recommended value)")
    p.add_argument("--log_per_image_loss", action="store_true", help="detect problem images (Problem Images window)")
    p.add_argument("--per_image_lr", action="store_true", help="scale each step by the image's loss-watch multiplier")
    p.add_argument("--auto_recaption", action="store_true", help="recaption stuck images (needs --captioner)")
    p.add_argument("--warmup_look_outliers", action="store_true", help="LR warm-up for Look Filter outliers")
    p.add_argument("--trigger_position", default="start", choices=("start", "end"))
    p.add_argument("--recaption_instruction", default=None)
    p.add_argument("--recaption_instruction_detailed", default=None)
    p.add_argument("--captioner", default=None,
                   help="auto-recaption's captioner: the Krea 2 Qwen3-VL-4B text encoder file (as the Captions tab)")
    p.add_argument("--lokr_factor", type=int, default=8, help="LoKR only: w1 is about factor x factor")
    p.add_argument("--network_alpha", type=float, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_epochs", type=int, default=16)
    p.add_argument("--save_every_n_epochs", type=int, default=1)
    p.add_argument("--save_state", action="store_true")
    p.add_argument("--save_state_on_train_end", action="store_true")
    p.add_argument("--keep_last_n_states", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--training_adapter", default=None, help="Frozen training adapter (off in previews and saves)")
    p.add_argument("--training_adapter_strength", type=float, default=1.0)
    p.add_argument("--context_lora_path", default=None)
    p.add_argument("--context_lora_strength", type=float, default=1.0)
    p.add_argument("--speed_lora", default=None,
                   help="The family's preview speed LoRA (description.preview_speed_lora): previews only")
    p.add_argument("--min_timestep", type=float, default=0.0, help="Noise band floor (0-1)")
    p.add_argument("--max_timestep", type=float, default=1.0, help="Noise band ceiling (0-1)")
    p.add_argument("--vae", default=None)
    p.add_argument("--text_encoder", default=None)
    p.add_argument("--sample_prompts", default=None, help="One prompt per line")
    p.add_argument("--sample_every_n_epochs", type=int, default=0)
    p.add_argument("--sample_width", type=int, default=None)
    p.add_argument("--sample_height", type=int, default=None)
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sample_cfg_scale", type=float, default=None)
    p.add_argument("--sample_negative", default=None)
    p.add_argument("--sample_at_first", action="store_true")
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--sample_reference", default=None, help="Edit previews: the photo every preview prompt edits")
    p.add_argument("--slider_pairs", action="store_true",
                   help="Slider from image pairs: each training image is the +1 pole, its control_directory pair "
                        "(same file name) the -1 pole")
    p.add_argument("--slider_diff_weight", type=float, default=1.0,
                   help="Image-pair slider: 0 = plain loss, 1 = concentrate where the two images differ")
    p.add_argument("--slider_prompts", nargs=3, default=None, metavar=("NEUTRAL", "POSITIVE", "NEGATIVE"),
                   help="Slider from three prompts, no images (needs --text_encoder and --vae)")
    p.add_argument("--slider_guidance", type=float, default=3.0, help="Prompt-pair slider: how hard to push")
    p.add_argument("--slider_bank", type=int, default=16, help="Prompt-pair slider: practice images to render")
    p.add_argument("--slider_bank_res", type=int, default=768, help="Prompt-pair slider: practice image size")
    for k in ("title", "author", "description", "license", "tags", "trigger_phrase", "thumbnail"):
        p.add_argument(f"--metadata_{k}", default=None)
    p.add_argument("--trigger_word", default=None, help="Recorded as the trigger phrase when none is given")
    p.add_argument("--resume", default=None)
    p.add_argument("--adaptive_lr", action="store_true")
    p.add_argument("--adaptive_lr_min", type=float, default=1e-4)
    p.add_argument("--adaptive_lr_max", type=float, default=2e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--optimizer_type", default="adamw")
    p.add_argument("--optimizer_args", default="")
    p.add_argument("--lr_scheduler", default="constant",
                   choices=["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear",
                            "polynomial"])
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_scheduler_num_cycles", type=int, default=1)
    p.add_argument("--lr_scheduler_power", type=float, default=1.0)
    return p


def main():
    if (sys.platform != "win32" and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    logging.basicConfig(level=logging.INFO)
    a = setup_parser().parse_args()
    prompts = None
    if a.sample_prompts and os.path.exists(a.sample_prompts):
        with open(a.sample_prompts, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    train_family(
        a.family, a.dit, a.dataset_config, a.output_dir, a.output_name, precision=a.precision,
        blocks_to_swap=a.blocks_to_swap, speed_lora_strength=a.speed_lora_strength, network_type=a.network_type, lokr_factor=a.lokr_factor,
        log_per_image_loss=a.log_per_image_loss, per_image_lr=a.per_image_lr, auto_recaption=a.auto_recaption,
        warmup_look_outliers=a.warmup_look_outliers, trigger_word=a.trigger_word, trigger_position=a.trigger_position,
        recaption_instruction=a.recaption_instruction,
        recaption_instruction_detailed=a.recaption_instruction_detailed, captioner=a.captioner,
        network_dim=a.network_dim, network_alpha=a.network_alpha, learning_rate=a.learning_rate,
        max_train_epochs=a.max_train_epochs, save_every_n_epochs=a.save_every_n_epochs, save_state=a.save_state,
        save_state_on_train_end=a.save_state_on_train_end, keep_last_n_states=a.keep_last_n_states, seed=a.seed,
        training_adapter=a.training_adapter, training_adapter_strength=a.training_adapter_strength,
        context_lora_path=a.context_lora_path, context_lora_strength=a.context_lora_strength,
        min_timestep=a.min_timestep, max_timestep=a.max_timestep, speed_lora=a.speed_lora,
        vae_path=a.vae, te_path=a.text_encoder,
        sample_prompts=prompts, sample_every_n_epochs=a.sample_every_n_epochs, sample_width=a.sample_width,
        sample_height=a.sample_height, sample_steps=a.sample_steps, sample_cfg_scale=a.sample_cfg_scale,
        sample_negative=a.sample_negative, sample_at_first=a.sample_at_first, sample_seed=a.sample_seed,
        sample_reference=[a.sample_reference] if a.sample_reference else None,
        slider_pairs=a.slider_pairs, slider_diff_weight=a.slider_diff_weight, slider_prompts=a.slider_prompts,
        slider_guidance=a.slider_guidance, slider_bank=a.slider_bank, slider_bank_res=a.slider_bank_res,
        metadata_title=a.metadata_title, metadata_author=a.metadata_author,
        metadata_description=a.metadata_description, metadata_license=a.metadata_license,
        metadata_tags=a.metadata_tags, metadata_trigger_phrase=a.metadata_trigger_phrase or a.trigger_word,
        metadata_thumbnail=a.metadata_thumbnail, resume_state_dir=a.resume, adaptive_lr=a.adaptive_lr,
        adaptive_lr_min=a.adaptive_lr_min, adaptive_lr_max=a.adaptive_lr_max, max_grad_norm=a.max_grad_norm,
        ema_decay=a.ema_decay, optimizer_type=a.optimizer_type, optimizer_args=a.optimizer_args,
        lr_scheduler=a.lr_scheduler, lr_warmup_steps=a.lr_warmup_steps,
        lr_scheduler_num_cycles=a.lr_scheduler_num_cycles, lr_scheduler_power=a.lr_scheduler_power)


if __name__ == "__main__":
    main()
