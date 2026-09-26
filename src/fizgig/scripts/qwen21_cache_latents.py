"""Cache Qwen Image 2.1 VAE latents (64-ch, 16x).

    python src/fizgig/scripts/qwen21_cache_latents.py --dataset_config config.toml --vae path/to/diffusion_pytorch_model.safetensors
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.dataset.config import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from fizgig.scripts.cache_latents import encode_datasets  # generic dataset iteration / stale-cleanup
from fizgig.qwen_image21.caching import encode_and_save_latents
from fizgig.qwen_image21.vae import load_qwen21_vae
from fizgig.training.metadata import ARCHITECTURE_QWEN21

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache Qwen Image 2.1 VAE latents")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config .toml file")
    parser.add_argument("--vae", type=str, required=True, help="Qwen Image 2.1 VAE, diffusers format (vae/diffusion_pytorch_model.safetensors)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="Keep stale cache files")
    return parser


def main():
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_gen = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Loading dataset config from {args.dataset_config}")
    user_config = load_user_config(args.dataset_config)
    blueprint = blueprint_gen.generate(user_config, args, architecture=ARCHITECTURE_QWEN21)
    datasets = generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets

    logger.info(f"Loading Qwen Image 2.1 VAE from {args.vae}")
    vae = load_qwen21_vae(args.vae, device=device)

    encode_datasets(datasets, lambda batch: encode_and_save_latents(vae, batch), args)


if __name__ == "__main__":
    if sys.platform == "linux" and os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm":
        from fizgig.rocm.cache_exit import run_cache_main

        run_cache_main(main)
    else:
        main()
