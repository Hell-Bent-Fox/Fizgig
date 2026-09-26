"""Qwen Image 2.1 caching on Fizgig's dataset framework (ItemInfo + safetensors cache files).

* Latents: RGB images get an opaque alpha channel (the VAE is RGBA), are encoded and normalised
  ((z - mean) / std, the checkpoint's latents_mean/std) and stored as `latent_{h}x{w}` (64, h, w) at
  pixel/16 - the same key the dataset layer maps to `latents`.
* Text: the Qwen3-VL-8B last-layer states before the final RMSNorm, system-turn tokens dropped
  (embedder.Qwen21TextEncoder), stored as `hidden_states` (L, 4096) with an all-True `attention_mask`.
  Lengths vary per caption, so the trainer runs batch size 1.
"""
import logging
import os
from typing import List

import torch
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import ItemInfo, dtype_to_str
from fizgig.qwen_image21 import sampling as S

ARCHITECTURE_QWEN21 = "qwenimage21"   # interim: moves to the family driver (standard layer)
logger = logging.getLogger(__name__)


def save_latent_cache(item_info: ItemInfo, latent: torch.Tensor) -> None:
    assert latent.dim() == 3, f"latent must be 3-D (C,H,W), got {tuple(latent.shape)}"
    _, H, W = latent.shape
    value = latent.detach().cpu().contiguous()
    if torch.isnan(value).any():
        logger.warning(f"NaN in latent for {item_info.item_key} - replaced with 0")
        value[torch.isnan(value)] = 0
    os.makedirs(os.path.dirname(item_info.latent_cache_path), exist_ok=True)
    save_file({f"latent_{H}x{W}": value}, item_info.latent_cache_path, metadata={
        "architecture": ARCHITECTURE_QWEN21, "width": str(item_info.original_size[0]),
        "height": str(item_info.original_size[1]), "dtype": dtype_to_str(value.dtype), "format_version": "1.0.0"})


def save_text_cache(item_info: ItemInfo, hidden_states: torch.Tensor) -> None:
    value = hidden_states.detach().cpu().contiguous()
    if torch.isnan(value).any():
        logger.warning(f"NaN in hidden_states for {item_info.item_key} - replaced with 0")
        value[torch.isnan(value)] = 0
    os.makedirs(os.path.dirname(item_info.text_encoder_output_cache_path), exist_ok=True)
    save_file({"hidden_states": value, "attention_mask": torch.ones(value.shape[0], dtype=torch.bool)},
              item_info.text_encoder_output_cache_path, metadata={
                  "architecture": ARCHITECTURE_QWEN21, "caption1": item_info.caption,
                  "dtype": dtype_to_str(value.dtype), "format_version": "1.0.0"})


@torch.no_grad()
def encode_and_save_latents(vae, batch: List[ItemInfo]) -> None:
    arrays = []
    for item in batch:
        content = item.content[0] if isinstance(item.content, list) else item.content  # (H, W, C) uint8
        arrays.append(torch.from_numpy(content[..., :3]))
    x = torch.stack(arrays).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    dev, dt = next(vae.parameters()).device, next(vae.parameters()).dtype
    latents = S.encode_image(vae, x.to(dev, dt))                   # (B, 64, h, w), normalised
    for item, z in zip(batch, latents):
        logger.info(f"latent cache: {item.item_key} -> {tuple(z.shape)}")
        save_latent_cache(item, z.to(torch.bfloat16))


@torch.no_grad()
def encode_and_save_text(encoder, batch: List[ItemInfo]) -> None:
    for item, h in zip(batch, encoder.encode([item.caption for item in batch])):
        save_text_cache(item, h.to(torch.bfloat16))
