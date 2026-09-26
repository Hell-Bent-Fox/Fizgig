"""Qwen Image 2.1 LoRA trainer.

A plain Qwen 2.1 LoRA is unstable: in Fizgig's lab it collapsed to texture at lr 5e-4 (step ~300) and
wobbled at 1e-4, and with the same Adaptive LR a no-adapter run fell to 37 likeness at step 2000.
Training runs with a frozen TRAINING ADAPTER active (Fizgig's own, trained on filtered photographs plus
the model's own renders); it absorbs the drift so the LoRA learns the subject. The adapter is switched
off for previews and is never part of the saved LoRA, so the file works on the plain base model.
Measured (26 Sep 2026, 170-image face set, Adaptive LR 1e-4..2e-4, 3000 steps): portrait likeness
77.4 with Fizgig's adapter, 76.4 with SimpleTuner's assistant v2, 55.8 with none; Peter judged the
Fizgig adapter's detail far higher.

Pipeline (GUI): qwen21_cache_latents -> qwen21_cache_text -> qwen21_train. Batch size 1 (caption
lengths vary). Base in bf16 (~14.2 GB resident).
"""
import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from multiprocessing import Value

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint,
                                   load_user_config)
from fizgig.krea2.trainer import AdaptiveLR
from fizgig.qwen_image21 import sampling as S
from fizgig.qwen_image21.lora import QwenLoRA
from fizgig.training.metadata import (build_metadata, latest_sample_image, resolve_title,
                                      thumbnail_data_uri)
from fizgig.training.train_utils import LossRecorder, prune_state_dirs

ARCHITECTURE_QWEN21 = "qwenimage21"   # interim: moves to the family driver (standard layer)
logger = logging.getLogger(__name__)

ADAPTER = "training_adapter"
CONTEXT = "context"


class _Collator:
    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def sample_sigma(n_tokens, gen, min_t=0.0, max_t=1.0):
    """Logit-normal t with the checkpoint's resolution-dependent exponential shift (the sampler's mu),
    optionally restricted to a [min_t, max_t] band."""
    t = torch.sigmoid(torch.randn(1, generator=gen)).item()
    mu = S.calculate_mu(n_tokens)
    sig = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
    return min_t + (max_t - min_t) * sig


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


def _save_state(output_dir, output_name, net, optimizer, *, epoch, global_step, extra=None, ema=None):
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
        json.dump({"epoch": epoch, "global_step": global_step, "architecture": ARCHITECTURE_QWEN21, **(extra or {})}, f)
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _load_state(state_dir, net, optimizer, device):
    for need in ("lora.safetensors", "training_state.json"):
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


@torch.no_grad()
def _render_previews(dit, net, vae, encoded, out_dir, epoch, *, output_name, steps, cfg, neg, width, height,
                     seed, ema=None, device="cuda"):
    """Previews on the live model with the training adapter OFF (the deployment setup)."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    from PIL import Image
    net.set_enabled(ADAPTER, False)
    if ema is not None:
        ema.swap_in()
    was_training = dit.training
    dit.eval()
    paths = []
    try:
        for i, emb in enumerate(encoded):
            x = S.sample(dit, emb, height, width, steps=steps, seed=seed + i, cfg=cfg, neg_emb=neg, device=device)
            img = S.decode(vae, x, height, width)[0, :3].float().clamp(-1, 1)
            p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
            Image.fromarray(((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)).save(p)
            paths.append(p)
    finally:
        if ema is not None:
            ema.swap_out()
        net.set_enabled(ADAPTER, True)
        dit.train(was_training)
    logger.info(f"[sample] epoch {epoch}: {len(paths)} preview(s) -> {out_dir}")
    return paths


def train_qwen21(dit_path, dataset_config, output_dir, output_name, *, network_dim=32, network_alpha=32,
                 learning_rate=1e-4, max_train_epochs=16, save_every_n_epochs=1, save_state=False,
                 save_state_on_train_end=False, keep_last_n_states=2, seed=42,
                 training_adapter=None, training_adapter_strength=1.0,
                 context_lora_path=None, context_lora_strength=1.0,
                 min_timestep=0.0, max_timestep=1.0,
                 vae_path=None, te_path=None, sample_prompts=None, sample_every_n_epochs=0, sample_width=1024,
                 sample_height=1024, sample_steps=25, sample_cfg_scale=1.0, sample_negative=None,
                 sample_at_first=False, sample_seed=42,
                 metadata_title=None, metadata_author=None, metadata_description=None, metadata_license=None,
                 metadata_tags=None, metadata_trigger_phrase=None, metadata_thumbnail=None,
                 resume_state_dir=None, adaptive_lr=False, adaptive_lr_min=1e-4, adaptive_lr_max=2e-4,
                 max_grad_norm=1.0, ema_decay=0.0, optimizer_type="adamw", optimizer_args="",
                 lr_scheduler="constant", lr_warmup_steps=0, lr_scheduler_num_cycles=1, lr_scheduler_power=1.0,
                 gradient_checkpointing=True):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ---- data ------------------------------------------------------------------------------------
    shared_epoch = Value("i", 0)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        load_user_config(dataset_config), argparse.Namespace(), architecture=ARCHITECTURE_QWEN21)
    group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True, num_timestep_buckets=None,
                                                shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items - run qwen21_cache_latents and qwen21_cache_text first.")
    for ds in group.datasets:
        if getattr(ds, "batch_size", 1) != 1:
            raise RuntimeError("Qwen Image 2.1 trains at batch size 1 (caption lengths differ per image). "
                               "Set Batch Size to 1.")
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=_Collator(shared_epoch, group), num_workers=0)
    steps_per_epoch = len(loader)
    logger.info(f"Qwen Image 2.1 training: {group.num_train_items} items, {max_train_epochs} epochs, "
                f"{steps_per_epoch} steps/epoch")

    # ---- previews: encode prompts once, keep the VAE ---------------------------------------------
    encoded = neg = vae = None
    sample_dir = os.path.join(output_dir, "sample")
    if sample_prompts and sample_every_n_epochs and te_path and vae_path:
        from fizgig.qwen_image21.embedder import Qwen21TextEncoder
        logger.info("[sample] encoding %d preview prompt(s) with Qwen3-VL-8B", len(sample_prompts))
        te = Qwen21TextEncoder(te_path)
        encoded = te.encode(sample_prompts)
        if sample_cfg_scale > 1.0:
            neg = te.encode([sample_negative or ""])[0]
        te.unload()
        del te
        torch.cuda.empty_cache()
    elif sample_prompts and sample_every_n_epochs:
        logger.warning("[sample] previews need the text encoder and VAE paths - previews are off for this run")
    if vae_path and encoded is not None:
        from fizgig.qwen_image21.vae import load_qwen21_vae
        vae = load_qwen21_vae(vae_path, device=device)

    # ---- model ----------------------------------------------------------------------------------
    from fizgig.qwen_image21.model import load_qwen21_dit
    logger.info(f"Loading Qwen Image 2.1 DiT from {dit_path}")
    dit = load_qwen21_dit(dit_path, device=device).eval().requires_grad_(False)
    if gradient_checkpointing:
        dit.enable_gradient_checkpointing(True)
    net = QwenLoRA(dit)
    if training_adapter:
        n = net.add_file(training_adapter, ADAPTER, training_adapter_strength)
        if n == 0:
            raise RuntimeError(f"Training adapter {training_adapter} matched no Qwen Image 2.1 modules.")
        logger.info(f"[adapter] training adapter ON ({n} Linears, strength {training_adapter_strength:g}) - frozen, "
                    f"off in previews, not saved into the LoRA")
    else:
        logger.warning("[adapter] NO training adapter - plain Qwen 2.1 LoRAs are unstable (collapse at high LR, "
                       "wobble at 1e-4). Expect weaker, less stable results.")
    if context_lora_path:
        n = net.add_file(context_lora_path, CONTEXT, context_lora_strength)
        logger.info(f"[context] {os.path.basename(context_lora_path)} frozen + active at {context_lora_strength:g} "
                    f"({n} Linears)")
    net.add_trainable(network_dim, network_alpha)
    params = net.parameters()
    logger.info(f"LoRA rank {network_dim} alpha {network_alpha:g}: {len(net.trainable_modules())} modules, "
                f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable params")

    from fizgig.training.optimizers import create_optimizer
    if adaptive_lr:
        learning_rate = math.sqrt(adaptive_lr_min * adaptive_lr_max)
        logger.info(f"[adaptive_lr] ENABLED - start_lr={learning_rate:.3e} min_lr={adaptive_lr_min:.3e} "
                    f"max_lr={adaptive_lr_max:.3e} (the Learning Rate box is ignored)")
    optimizer, opt_label = create_optimizer(optimizer_type, params, learning_rate, optimizer_args)
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
    scheduler = None
    if not adaptive:
        scheduler = _step_scheduler(optimizer, lr_scheduler, lr_warmup_steps, steps_per_epoch * max_train_epochs,
                                    lr_scheduler_num_cycles, lr_scheduler_power)
        for _ in range(global_step):
            scheduler.step()

    last_prompt = [None]

    def metadata(epoch):
        thumb = None if (metadata_thumbnail or "").lower() in ("off", "none") else (
            metadata_thumbnail or latest_sample_image(output_dir))
        md = build_metadata(None, ARCHITECTURE_QWEN21, time.time(),
                            title=metadata_title if metadata_title is not None else resolve_title(output_name, metadata_trigger_phrase),
                            author=metadata_author,
                            description=metadata_description if metadata_description is not None else last_prompt[0],
                            license=metadata_license, tags=metadata_tags, trigger_phrase=metadata_trigger_phrase,
                            thumbnail=thumbnail_data_uri(thumb))
        md.update({"ss_network_module": "fizgig.qwen_image21 (lora, attn+mlp)", "ss_network_dim": str(network_dim),
                   "ss_network_alpha": str(network_alpha), "ss_architecture": ARCHITECTURE_QWEN21,
                   "ss_epoch": str(epoch), "ss_optimizer": opt_label, "ss_learning_rate": f"{learning_rate:g}",
                   "ss_training_adapter": os.path.basename(training_adapter) if training_adapter else "none"})
        if context_lora_path:
            md.update({"ss_context_lora": os.path.basename(context_lora_path),
                       "ss_context_lora_strength": str(context_lora_strength)})
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
        _render_previews(dit, net, vae, encoded, sample_dir, epoch, output_name=output_name, steps=sample_steps,
                         cfg=sample_cfg_scale, neg=neg, width=sample_width, height=sample_height, seed=sample_seed,
                         ema=ema)
        last_prompt[0] = sample_prompts[-1] if sample_prompts else None

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
        t0 = time.time()
        for i, batch in enumerate(loader):
            z = batch["latents"].to(device, torch.float32)                  # (1, 64, h, w)
            h, w = z.shape[-2:]
            n = h * w
            x0 = S.pack(z)
            noise = torch.randn(x0.shape, generator=gen).to(device)
            sig = sample_sigma(n, gen, min_timestep, max_timestep)
            xt = (1 - sig) * x0 + sig * noise
            enc, img_mask, enc_mask = S.model_inputs(batch["hidden_states"][0], n, device)
            t = torch.tensor([sig], device=device, dtype=torch.bfloat16)
            pred = dit(xt.to(torch.bfloat16), enc.to(torch.bfloat16), t, [[(1, h, w)]], img_mask, enc_mask)[:, -n:]
            loss = F.mse_loss(pred.float(), noise - x0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update()
            global_step += 1
            recorder.add(epoch=epoch, step=i, loss=loss.item())
            progress.set_postfix(avr_loss=f"{recorder.moving_average:.4f}", refresh=False)
            progress.update(1)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"epoch {epoch + 1}/{max_train_epochs}  avr_loss={recorder.moving_average:.4f}  step={global_step}  "
                    f"{(time.time() - t0) / max(1, steps_per_epoch):.2f}s/step  lr={lr_now:.3e}")
        if adaptive:
            adaptive.epoch_boundary(epoch, recorder.moving_average, net.trainable_modules(), optimizer)

        done = epoch + 1
        if save_every_n_epochs and done % save_every_n_epochs == 0 and done < max_train_epochs:
            save_lora(os.path.join(output_dir, f"{output_name}-{done:06d}.safetensors"), done)
        state_saved = False
        if save_state and save_every_n_epochs and done % save_every_n_epochs == 0 and done < max_train_epochs:
            _save_state(output_dir, output_name, net, optimizer, epoch=done, global_step=global_step, ema=ema,
                        extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
            prune_state_dirs(output_dir, output_name, keep_last_n_states)
            state_saved = True
        if sample_every_n_epochs and done % sample_every_n_epochs == 0:
            previews(done)
        if os.path.exists(pause_flag) and done < max_train_epochs:
            if not state_saved:
                logger.info(f"[pause] requested - saving state at epoch {done} and exiting cleanly")
                _save_state(output_dir, output_name, net, optimizer, epoch=done, global_step=global_step, ema=ema,
                            extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
            else:
                logger.info(f"[pause] requested - state for epoch {done} already saved; exiting cleanly")
            progress.close()
            sys.exit(0)

    progress.close()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    save_lora(final, max_train_epochs)
    if save_state_on_train_end:
        _save_state(output_dir, output_name, net, optimizer, epoch=max_train_epochs, global_step=global_step, ema=ema,
                    extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
    logger.info(f"Training complete -> {final}")
    return final
