"""Standard-layer caching for any described family: VAE latents or text conditioning, through the family driver.

    python src/fizgig/families/cache.py --family qwen_image21 --stage latents --dataset_config X.toml --model VAE
    python src/fizgig/families/cache.py --family qwen_image21 --stage text    --dataset_config X.toml --model TE

Uses Fizgig's dataset framework (bucketing, stale-cache cleanup, --skip_existing) exactly like the per-family
scripts. Latents are stored as `latent_{h}x{w}`; conditioning as `cond__<driver key>` (passed through verbatim by
the dataset loader and handed back to the driver as the same dict).
"""
import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from safetensors.torch import save_file  # noqa: E402

from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,  # noqa: E402
                                   generate_dataset_group_by_blueprint, load_user_config)
from fizgig.dataset.image_dataset import dtype_to_str  # noqa: E402
from fizgig.families.registry import get  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
FORMAT_VERSION = "1.0.0"


def _clean(t, what, key):
    t = t.detach().cpu().contiguous()
    if t.is_floating_point() and torch.isnan(t).any():
        logger.warning(f"NaN in {what} for {key} - replaced with 0")
        t[torch.isnan(t)] = 0
    return t


def save_latents(desc, item, latent):
    _, h, w = latent.shape
    os.makedirs(os.path.dirname(item.latent_cache_path), exist_ok=True)
    save_file({f"latent_{h}x{w}": _clean(latent, "latent", item.item_key)}, item.latent_cache_path, metadata={
        "architecture": desc.arch_id, "width": str(item.original_size[0]), "height": str(item.original_size[1]),
        "dtype": dtype_to_str(latent.dtype), "format_version": FORMAT_VERSION})


def save_cond(desc, item, cond):
    os.makedirs(os.path.dirname(item.text_encoder_output_cache_path), exist_ok=True)
    save_file({f"cond__{k}": _clean(v, k, item.item_key) for k, v in cond.items()},
              item.text_encoder_output_cache_path,
              metadata={"architecture": desc.arch_id, "caption1": item.caption, "format_version": FORMAT_VERSION})


def main():
    p = argparse.ArgumentParser(description="Cache latents or text conditioning for a described model family")
    p.add_argument("--family", required=True, help="family key, e.g. qwen_image21")
    p.add_argument("--stage", required=True, choices=["latents", "text"])
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--model", required=True, help="the VAE (latents) or text encoder (text) file")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--keep_cache", action="store_true")
    args = p.parse_args()

    desc = get(args.family)
    if desc is None or not desc.training_ready:
        raise SystemExit(f"unknown or untrainable family {args.family!r}")
    driver = desc.load_driver()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(load_user_config(args.dataset_config), args,
                                                               architecture=desc.arch_id)
    datasets = generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets

    if args.stage == "latents":
        from fizgig.scripts.cache_latents import encode_datasets
        vae = driver.load_vae(args.model, device)

        def encode(batch):
            imgs = [(it.content[0] if isinstance(it.content, list) else it.content) for it in batch]
            for it, z in zip(batch, driver.encode_images(vae, imgs)):
                logger.info(f"latent cache: {it.item_key} -> {tuple(z.shape)}")
                save_latents(desc, it, z)
        encode_datasets(datasets, encode, args)
    else:
        from fizgig.scripts.cache_text import post_process, prepare_cache_files_and_paths, process_batches
        if args.batch_size is None:
            args.batch_size = 8
        files, paths = prepare_cache_files_and_paths(datasets)
        te = driver.load_text_encoder(args.model, device)

        def encode(batch):
            for it, c in zip(batch, driver.encode_text(te, [it.caption for it in batch])):
                save_cond(desc, it, c)
        process_batches(args, datasets, files, paths, encode)
        driver.unload_text_encoder(te)
        post_process(datasets, files, paths, args.keep_cache)


if __name__ == "__main__":
    main()
