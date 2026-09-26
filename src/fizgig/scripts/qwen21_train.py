"""Qwen Image 2.1 LoRA training CLI (wraps fizgig.qwen_image21.trainer.train_qwen21).

The GUI drives the full run as three steps: qwen21_cache_latents -> qwen21_cache_text -> qwen21_train.
Trains with Fizgig's frozen training adapter active (--training_adapter); previews run adapter-off.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Same process-level settings as the other trainers (see krea2_train.py for the measurements).
if (sys.platform != "win32" and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from fizgig.qwen_image21.trainer import train_qwen21  # noqa: E402
from fizgig.training.optimizers import available_optimizers  # noqa: E402

logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen Image 2.1 LoRA training")
    p.add_argument("--dit", required=True, help="Qwen Image 2.1 DiT (qwen_image_2.1_bf16.safetensors)")
    p.add_argument("--dataset_config", required=True, help="Dataset .toml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--network_dim", type=int, default=32)
    p.add_argument("--network_alpha", type=float, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_epochs", type=int, default=16)
    p.add_argument("--save_every_n_epochs", type=int, default=1)
    p.add_argument("--save_state", action="store_true", help="Write a resumable <name>-NNNNNN-state/ at each checkpoint")
    p.add_argument("--save_state_on_train_end", action="store_true")
    p.add_argument("--keep_last_n_states", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--training_adapter", default=None,
                   help="Frozen training adapter, active during training and off in previews/saves. Without it "
                        "Qwen 2.1 LoRAs are unstable.")
    p.add_argument("--training_adapter_strength", type=float, default=1.0)
    p.add_argument("--context_lora_path", default=None, help="Frozen + active context LoRA on the base")
    p.add_argument("--context_lora_strength", type=float, default=1.0)
    p.add_argument("--min_timestep", type=float, default=0.0, help="Noise band floor (0-1)")
    p.add_argument("--max_timestep", type=float, default=1.0, help="Noise band ceiling (0-1)")
    p.add_argument("--vae", default=None, help="Qwen Image 2.1 VAE (diffusers format) for preview decode")
    p.add_argument("--text_encoder", default=None, help="Qwen3-VL-8B for preview prompt encode")
    p.add_argument("--sample_prompts", default=None, help="Sample-prompts file (one prompt per line)")
    p.add_argument("--sample_every_n_epochs", type=int, default=0)
    p.add_argument("--sample_width", type=int, default=1024)
    p.add_argument("--sample_height", type=int, default=1024)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--sample_cfg_scale", type=float, default=1.0)
    p.add_argument("--sample_negative", default=None)
    p.add_argument("--sample_at_first", action="store_true")
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--metadata_title", default=None)
    p.add_argument("--metadata_author", default=None)
    p.add_argument("--metadata_description", default=None)
    p.add_argument("--metadata_license", default=None)
    p.add_argument("--metadata_tags", default=None)
    p.add_argument("--metadata_trigger_phrase", default=None)
    p.add_argument("--metadata_thumbnail", default=None)
    p.add_argument("--trigger_word", default=None, help="Recorded as the trigger phrase when none is given")
    p.add_argument("--resume", default=None, help="Path to a <name>-NNNNNN-state dir")
    p.add_argument("--adaptive_lr", action="store_true", help="Bi-directional plateau LR tracker")
    p.add_argument("--adaptive_lr_min", type=float, default=1e-4)
    p.add_argument("--adaptive_lr_max", type=float, default=2e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--optimizer_type", default="adamw",
                   help="Optimizer family. Available here: " + ", ".join(available_optimizers()))
    p.add_argument("--optimizer_args", default="")
    p.add_argument("--lr_scheduler", default="constant",
                   choices=["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear",
                            "polynomial"])
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_scheduler_num_cycles", type=int, default=1)
    p.add_argument("--lr_scheduler_power", type=float, default=1.0)
    return p


def main():
    args = setup_parser().parse_args()
    prompts = None
    if args.sample_prompts and os.path.exists(args.sample_prompts):
        with open(args.sample_prompts, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    train_qwen21(
        args.dit, args.dataset_config, args.output_dir, args.output_name,
        network_dim=args.network_dim, network_alpha=args.network_alpha, learning_rate=args.learning_rate,
        max_train_epochs=args.max_train_epochs, save_every_n_epochs=args.save_every_n_epochs,
        save_state=args.save_state, save_state_on_train_end=args.save_state_on_train_end,
        keep_last_n_states=args.keep_last_n_states, seed=args.seed,
        training_adapter=args.training_adapter, training_adapter_strength=args.training_adapter_strength,
        context_lora_path=args.context_lora_path, context_lora_strength=args.context_lora_strength,
        min_timestep=args.min_timestep, max_timestep=args.max_timestep,
        vae_path=args.vae, te_path=args.text_encoder, sample_prompts=prompts,
        sample_every_n_epochs=args.sample_every_n_epochs, sample_width=args.sample_width,
        sample_height=args.sample_height, sample_steps=args.sample_steps, sample_cfg_scale=args.sample_cfg_scale,
        sample_negative=args.sample_negative, sample_at_first=args.sample_at_first, sample_seed=args.sample_seed,
        metadata_title=args.metadata_title, metadata_author=args.metadata_author,
        metadata_description=args.metadata_description, metadata_license=args.metadata_license,
        metadata_tags=args.metadata_tags, metadata_trigger_phrase=args.metadata_trigger_phrase or args.trigger_word,
        metadata_thumbnail=args.metadata_thumbnail, resume_state_dir=args.resume,
        adaptive_lr=args.adaptive_lr, adaptive_lr_min=args.adaptive_lr_min, adaptive_lr_max=args.adaptive_lr_max,
        max_grad_norm=args.max_grad_norm, ema_decay=args.ema_decay, optimizer_type=args.optimizer_type,
        optimizer_args=args.optimizer_args, lr_scheduler=args.lr_scheduler, lr_warmup_steps=args.lr_warmup_steps,
        lr_scheduler_num_cycles=args.lr_scheduler_num_cycles, lr_scheduler_power=args.lr_scheduler_power,
    )


if __name__ == "__main__":
    main()
