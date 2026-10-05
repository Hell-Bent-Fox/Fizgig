"""SDXL through the standard layer: any SDXL checkpoint as one .safetensors file (Juggernaut XL by default; Illustrious,
Pony, RealVis and the rest load the same way). The checkpoint carries the UNet, VAE and both CLIP text encoders, so
the VAE and text-encoder rows are optional overrides (ModelFile.inside). Driver: sdxl/driver.py.
"""
from fizgig.families.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

_JUGG = "RunDiffusion/Juggernaut-XL-v9"
_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
_IN = ("IN04", "IN05", "IN07", "IN08")
_MID = ("MID",)
_OUT = ("OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05")


def _preset(rank, lr=1e-4, adaptive=None, epochs=20, mp="1.0"):
    lo, hi = adaptive or ("1e-4", "4e-4")
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": adaptive is not None, "ADAPTIVE_LR_MIN": lo, "ADAPTIVE_LR_MAX": hi,
        "OPTIMIZER_TYPE": "adamw8bit", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
        "DATASET_MEGAPIXELS": mp, "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "FAMILY_EMA": "Off",
        "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
        "KREA2_WARMUP_LOOK": False,
    }


SDXL = FamilyDescription(
    key="sdxl",
    arch_id="sdxl",
    display_name="SDXL",
    gui_label="SDXL (any checkpoint, experimental)",
    lora_name_suffix="sdxl",
    aliases=("sdxl", "stable-diffusion-xl"),
    experimental=True,

    model_files=(
        ModelFile("sdxl_checkpoint", "SDXL checkpoint", True, _JUGG, "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
                  7.11, role="dit",
                  hint="Any SDXL checkpoint (.safetensors, the single file ComfyUI loads). Juggernaut XL is a strong "
                       "photographic base; Illustrious, Pony, RealVis and other SDXL fine-tunes work the same way. "
                       "Train on the checkpoint you will use the LoRA with.",
                  download_label="Download Juggernaut XL v9",
                  download_note="~7.1 GB - RunDiffusion (Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors), "
                                "CreativeML OpenRAIL-M",
                  alt_repo="OnomaAIResearch/Illustrious-xl-early-release-v0", alt_path="Illustrious-XL-v0.1.safetensors",
                  alt_label="Download Illustrious XL v0.1", alt_note="~6.9 GB - anime / illustration base"),
        ModelFile("sdxl_vae", "VAE (optional)", False, "stabilityai/sdxl-vae", "sdxl_vae.safetensors", 0.33,
                  role="vae", inside="sdxl_checkpoint", fetch_optional=False,
                  hint="Leave empty to use the checkpoint's own VAE. Set it only to swap in a separate SDXL VAE."),
        ModelFile("sdxl_text_encoder", "Text encoders (optional)", False, role="text_encoder", inside="sdxl_checkpoint",
                  fetch_optional=False,
                  hint="Leave empty: both CLIP text encoders are read from the checkpoint. Set it to another SDXL "
                       "checkpoint to take its text encoders instead."),
    ),
    prefs_title="Model Paths (SDXL)",
    prefs_intro="One SDXL checkpoint is all SDXL needs: its VAE and text encoders come from the same file.",
    text_encoder_label="CLIP-L + CLIP-G",
    vae_label="SDXL VAE",

    latent_channels=4,                # unet/config.json in_channels 4
    spatial_factor=8,                 # vae/config.json: 4 downsampling blocks -> 1/8
    bucket_step=64,                   # SDXL's training buckets are multiples of 64
    native_megapixels=1.0,            # trained at 1024x1024 and its aspect buckets

    n_blocks=11,                      # the 11 attention modules (SDXL block-weight names IN04 ... OUT05)
    block_prefix="unet",              # unused: the driver's block_map names SDXL's own blocks
    block_note="SDXL's attention modules under their block-weight names: input 4, 5, 7, 8, the middle block and "
               "output 0-5 (the resnets and the outer input / output blocks carry no attention and are not trained).",
    extract_presets=(
        ("All Blocks", ()),
        ("Input blocks", tuple((b, 1.0) for b in _IN)),
        ("Middle + output blocks", tuple((b, 1.0) for b in _MID + _OUT)),
    ),
    workbench=("repair", "explorer", "profiler", "extract", "royale"),

    lora=LoRAFormat(
        key_template="lora_unet_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                       "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys on diffusers module names (lora_unet_down_blocks_1_attentions_0_...), which ComfyUI maps for "
             "SDXL. Community files in the LDM layout (lora_unet_input_blocks_4_1_...) load too (driver alias_flat); "
             "their text-encoder parts (lora_te1_ / lora_te2_) are not used.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet (diffusers keys -> lora_unet_ names)",
    ),

    driver="fizgig.sdxl.driver:SDXLDriver",
    modelspec_arch="stable-diffusion-xl-v1-base/lora",
    implementation="https://github.com/Stability-AI/generative-models",
    ema_default="Off",
    precisions=("bf16",),             # 2.6B UNet: 5.1 GB in bf16, fits training on 12 GB cards
    optimizers=("adamw8bit", "adamw"),
    network_types=("lora",),
    workbench_follows_samples=True,   # previews take the Samples tab's steps, CFG and negative
    helper_files=((_SDXL, ("model_index.json", "*/config.json", "tokenizer/*", "tokenizer_2/*", "scheduler/*")),),

    sampling=(
        SamplingSettings("Juggernaut (community)", steps=30, cfg=4.5, sampler="dpmpp_2m_sde", scheduler="karras",
                         options=(("sampler", "dpmpp_2m_sde_karras"),), negative_prompt=True,
                         note="DPM++ 2M SDE Karras, 30 steps, CFG 4-5, little or no negative: Fooocus's Juggernaut "
                              "default and the Civitai card. CFG above 6-7 turns skin waxy.",
                         source="lllyasviel/Fooocus presets/default.json; civarchive.com/models/133005; "
                                "rundiffusion.com/juggernaut-xl-rundiffusion-guide"),
        SamplingSettings("Euler (softer)", steps=30, cfg=4.5, sampler="euler", scheduler="normal",
                         options=(("sampler", "euler"),), negative_prompt=True,
                         note="Plain Euler, trailing spacing (starts at full noise): a softer look, less pore detail.",
                         source="rundiffusion.com/prompt-guide-for-juggernaut-xi-and-xii; "
                                "huggingface.co/RunDiffusion/Juggernaut-XL-v9/discussions/4"),
    ),
    preview_steps=30,
    preview_cfg=4.5,
    preview_negative="",               # Juggernaut's card and Fooocus: start with no negative
    preview_cfg_note="SDXL needs CFG: about 4 to 5 (Juggernaut's skin turns waxy above 6-7). Above 1 the negative "
                     "prompt applies.",
    preview_width=1024,
    preview_height=1024,

    presets=(
        ("✨ SDXL Standard (rank 16, adaptive LR)", _preset(16, adaptive=("1e-4", "4e-4"))),
        ("✨ SDXL Strong (rank 32, adaptive LR)", _preset(32, adaptive=("5e-5", "2e-4"))),
        ("✨ SDXL Style (rank 16, 1e-4)", _preset(16, lr=1e-4)),
    ),

    notes=(
        ("Captions: each CLIP reads 77 tokens (about 60 words); anything longer is cut off.",
         "tokenizer max_length 77"),
        ("Training: DDPM epsilon prediction on SDXL's scaled-linear schedule (betas 0.00085-0.012, 1000 steps), "
         "uniform timesteps, unweighted MSE; size conditioning (time_ids) from the bucket size with no crop.",
         f"{_SDXL} scheduler/scheduler_config.json"),
        ("The VAE runs in fp32: SDXL's VAE overflows in half precision.", "madebyollin/sdxl-vae-fp16-fix card"),
    ),
)
