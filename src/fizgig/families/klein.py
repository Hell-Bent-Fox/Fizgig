"""Klein 9B through the standard layer, beside the old Klein trainer ("Klein (driver)") until the two are shown to
train the same. The same model files and Preferences rows as Klein (shares_prefs_with), ComfyUI-compatible kohya keys
like the old trainer's, and its own cache files (arch_id klein9bdrv - a fresh cache, never mixed with the old one).
Facts are the old trainer's (src/fizgig/training/trainer.py, klein/model_utils.py, networks/lora_klein.py), cited per
value.
"""
from fizgig.families.description import FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

_BFL = "black-forest-labs"
_DOUBLE = ("img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
           "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2")

KLEIN = FamilyDescription(
    key="klein_driver",
    arch_id="klein9bdrv",
    display_name="Klein 9B",
    gui_label="Klein (driver)",
    lora_name_suffix="k9b",
    experimental=True,

    # the old Klein Preferences rows (lora_trainer_gui.py "Model Paths"): one set of paths for both trainers
    model_files=(
        ModelFile("base_dit", "DiT Base", True, f"{_BFL}/FLUX.2-klein-base-9b-fp8",
                  "flux-2-klein-base-9b-fp8.safetensors", 9.6,
                  "The undistilled base the LoRA trains on (BFL's fp8, or the bf16 from FLUX.2-klein-base-9B).",
                  role="dit"),
        ModelFile("vae", "VAE", True, f"{_BFL}/FLUX.2-dev", "ae.safetensors", 0.34,
                  "ae.safetensors from the repo root, not vae/diffusion_pytorch_model.safetensors.", role="vae"),
        ModelFile("text_encoder", "Text encoder", True, "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
                  "split_files/text_encoders/qwen_3_8b.safetensors", 16.4,
                  "Qwen3-8B. Used for caching only, then unloaded before training steps.", role="text_encoder"),
        ModelFile("distilled_dit", "DiT Distilled", False, f"{_BFL}/FLUX.2-klein-9b-fp8",
                  "flux-2-klein-9b-fp8.safetensors", 9.5,
                  "The 4-step model the workbench previews on.", role="preview_dit"),
    ),
    shares_prefs_with="klein",
    text_encoder_label="Qwen3-8B",
    vae_label="FLUX.2 AE",

    latent_channels=128,              # the AE's 32 channels, 2x2-packed into the channel axis
    spatial_factor=16,                # /8 encoder + the 2x2 pack (dataset/image_dataset.py LATENT_SPATIAL_FACTOR)
    bucket_step=16,                   # RESOLUTION_STEPS, the old Klein bucket grid
    image_channels=3,
    native_megapixels=1.0,

    n_blocks=32,                      # 8 double-stream + 24 single-stream blocks (Klein9BParams)
    block_prefix="double_blocks",
    block_note="Block ids follow the old Repair Studio (double_0-7, single_0-23), so its saved presets apply.",

    lora=LoRAFormat(
        key_template="lora_unet_double_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=_DOUBLE,
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys, as every Fizgig Klein LoRA: the double blocks' attention and MLP Linears and the single "
             "blocks' linear1 / linear2 (modulation and norms excluded); ComfyUI reads them.",
        source="src/fizgig/networks/lora_klein.py create_arch_network",
    ),

    driver="fizgig.klein.driver:KleinDriver",
    modelspec_arch="Flux.2-klein-9b",
    implementation="https://github.com/black-forest-labs/flux2",
    ema_default="Off",                # the doc's K10: available, off until an A/B says otherwise
    precisions=("bf16", "int8", "nf4"),
    auto_precisions=("int8", "nf4"),
    optimizers=("adamw8bit", "adamw"),
    network_types=("lora", "lokr"),
    edit_training=True,               # Klein is an edit model: references ride after the image tokens
    slider_training=True,
    preview_checkpoint_sampling=SamplingSettings(
        "Distilled 4-step", steps=4, cfg=1.0, sampler="euler", scheduler="simple",
        options=(("schedule", "simple"), ("shift", 2.02), ("guidance", 1.0)),
        note="Guidance-distilled: no CFG; ComfyUI's Euler simple schedule at shift 2.02.",
        source="src/fizgig/training/trainer.py sample_image_inference (use_distilled)"),

    sampling=(
        SamplingSettings("Base", steps=20, cfg=3.5, sampler="euler", scheduler="simple", negative_prompt=True,
                         note="Base has no guidance embed: only CFG steers it, and CFG 1 ignores the negative "
                              "prompt. Empirical-mu schedule.",
                         source="src/fizgig/klein/model_utils.py KLEIN_MODEL_INFO['klein-base-9b'], get_schedule"),
    ),
    preview_steps=20,
    preview_cfg=1.0,                  # the old SAMPLE_CFG_SCALE default
    preview_cfg_note="Base has no guidance embed, so CFG is the only guidance: 1 = none (the negative prompt is "
                     "ignored); about 3.5 to 4.5 gives guided previews.",
    preview_width=1024,
    preview_height=1024,
    notes=(
        ("Timesteps: flux2_shift - logit-normal, shift exp(mu) with mu 0.5 at 256 latent tokens to 1.15 at 4096, "
         "unweighted MSE on the flow-matching velocity.",
         "src/fizgig/training/trainer.py get_noisy_model_input_and_timesteps"),
    ),
)
