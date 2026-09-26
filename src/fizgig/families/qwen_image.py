"""Qwen Image 2.1: the first family described through FamilyDescription.

Facts from the phase-0 research (26 Sep 2026; full notes in the Desktop fizgig_family_descriptions
RESEARCH_qwen_image_2_1_*.md files). Sources are cited per value. Training entry points stay None until
the family is trained through its driver (qwen_image21/driver.py) by the generic cache + train entry points.
"""
from fizgig.families.description import (
    FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA,
)

_CARD = "https://huggingface.co/Qwen/Qwen-Image-2.1"
_COMFY = "Comfy-Org/Qwen-Image-2.1"
_VIGGLE = "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo"
_TEMPLATE = "Comfy-Org workflow_templates templates/image_qwen_image_2_1_t2i.json"
_REDDIT = "r/StableDiffusion 'Qwen Image 2.1 4 Steps Turbo Lora is here by Viggle' (community, Sep 2026)"

QWEN_IMAGE_21 = FamilyDescription(
    key="qwen_image21",
    arch_id="qwenimage21",
    display_name="Qwen Image 2.1",
    gui_label="Qwen Image 2.1 (experimental)",
    lora_name_suffix="qwen21",
    aliases=("qwen-image-2.1", "qwen21"),
    experimental=True,

    model_files=(
        ModelFile("qwen21_dit", "Qwen Image 2.1 DiT", True, _COMFY,
                  "diffusion_models/qwen_image_2.1_bf16.safetensors", 14.23,
                  "bf16 base for training. ComfyUI's single file ships the MLP pre-fused (gate_up).", role="dit"),
        ModelFile("qwen21_vae", "Qwen Image 2.1 VAE", True, "Qwen/Qwen-Image-2.1",
                  "vae/diffusion_pytorch_model.safetensors", 1.35,
                  "Its own VAE (64 latent channels, 16x, RGBA), the diffusers-format file from the official "
                  "repo. Not the Krea 2 / Qwen-Image VAE.",
                  local_name="qwen_image_2.1_vae_diffusers.safetensors", role="vae"),
        ModelFile("qwen21_text_encoder", "Qwen3-VL-8B text encoder", True, _COMFY,
                  "text_encoders/qwen3vl_8b_bf16.safetensors", 17.53,
                  "Used for caching only, then unloaded before training steps.", role="text_encoder"),
        ModelFile("qwen21_training_adapter", "Fizgig training adapter", False,
              "shootthesound/Fizgig-Qwen-Image-2.1-Training-Adapter",
              "fizgig_qwen_image_2.1_training_adapter.safetensors", 0.08,
              "Frozen during training, off in previews and saved LoRAs. Without it Qwen 2.1 LoRAs collapse or "
              "wobble; with it likeness was 77 vs 56 in Fizgig's A/B.", role="training_adapter"),
    ModelFile("qwen21_turbo_lora", "Viggle turbo LoRA (previews)", False,
                  "Viggle/Qwen-Image-2.1-viggle-turbo",
                  "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors", 0.68,
                  "Optional: fast in-training previews. Applied unmerged.", role="speed_lora"),
    ),
    text_encoder_label="Qwen3-VL-8B",
    vae_label="Qwen Image 2.1 VAE",

    latent_channels=64,               # vae/config.json z_dim 64
    spatial_factor=16,                # vae/config.json scale_factor_spatial 16
    bucket_step=32,                   # VAE 16 x 2x2 image-pad slot grouping (ai-toolkit bucket divisibility)
    image_channels=4,                 # RGBA in/out; RGB training images get an opaque alpha channel
    native_megapixels=4.0,            # "natively supports 2K" (GitHub README aspect table)

    n_blocks=32,                      # transformer/config.json num_layers 32, identical single-stream blocks
    block_prefix="transformer_blocks",
    block_note="Modulation is shared across all blocks (one global Linear), so per-block sliders act "
               "on attention and MLP only. No block map (style / identity) exists yet.",

    lora=LoRAFormat(
        key_template="transformer.transformer_blocks.{block}.{module}.{ab}.weight",
        down="lora_A", up="lora_B",
        block_modules=("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                       "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"),
        alpha_key="{prefix}.alpha",
        kohya=False,
        note="Approved exception to the kohya-key rule (Peter, 26 Sep 2026): ComfyUI maps "
             "gate_layer/proj onto the halves of its fused gate_up only for bare or transformer.-prefixed "
             "keys; lora_unet_ / diffusion_model. keys silently drop the MLP input on the fused checkpoint.",
        source="ComfyUI comfy/lora.py model_lora_keys_unet QwenImage branch (commit 6bfaacc67c); "
               "Viggle r128 header",
    ),

    driver="fizgig.qwen_image21.driver:QwenImage21Driver",
    modelspec_arch="Qwen-Image-2.1",
    training_adapter="qwen21_training_adapter",
    training_adapter_note=("Keeps Qwen 2.1 LoRA training stable: frozen at 1.0 for every training step, off for "
                           "previews and never in your saved file. Without it Qwen 2.1 LoRAs collapse or wobble "
                           "(likeness 77 with it vs 56 without in Fizgig's A/B)."),
    implementation="https://github.com/QwenLM/Qwen-Image",
    precisions=("bf16",),             # int8 / nf4 bases not built yet; bf16 needs ~24 GB+
    optimizers=("adamw", "adamw8bit"),

    sampling=(
        SamplingSettings("ComfyUI template", steps=25, cfg=1.0, sampler="euler", scheduler="simple",
                         note="1 MP default; model shift 0.69 (mu) built in, no ModelSampling node.",
                         source=_TEMPLATE),
        SamplingSettings("Official (diffusers)", steps=40, cfg=1.0, sampler="euler", scheduler="simple",
                         note="FlowMatch Euler, dynamic exponential shift (mu 0.5 at 256 tokens to 0.9 at 8192), "
                              "shift_terminal 0.02. 'Meant to be sampled without guidance.'",
                         source=f"{_CARD} + diffusers pipeline_qwenimage21.py"),
        SamplingSettings("Community quality", steps=40, cfg=3.0, sampler="euler", scheduler="simple",
                         negative_prompt=True,
                         note="Community claim: CFG ~3-3.5 with 30-50 steps is more coherent than CFG 1; "
                              "25 steps shows banding that is gone at 35-40.",
                         source=f"{_REDDIT}; HF discussion threads"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="Viggle turbo v0.2.1 (6-step)",
            repo="Viggle/Qwen-Image-2.1-viggle-turbo",
            file="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
            pairs_with="Qwen Image 2.1 base transformer",
            strength=1.0,
            settings=SamplingSettings("Viggle 6-step", steps=6, cfg=1.0, sampler="euler", scheduler="simple",
                                      sigmas=(1.0, 0.9375, 0.875, 0.75, 0.5, 0.25),
                                      note="shift_terminal must be null; 8 steps for small text. "
                                           "Add or remove steps only at the high-noise end.",
                                      source=_VIGGLE),
            load_unmerged=True,
            community_settings=(
                ("8-20 steps at strength 0.3-0.8 as a clean-up rather than a speed-up; CFG 1-2 works "
                 "when the strength is lowered", _REDDIT),
                ("strength ~0.3 removes the grid pattern almost completely", _REDDIT),
            ),
            caveats=("Grid pattern at 4 steps and full strength (community).",
                     "Weaker for editing than for text-to-image (community).",
                     "Merged loading keeps only part of it (LPIPS 0.093 merged vs 0.052 unmerged)."),
            source=_VIGGLE,
        ),
    ),
    # Previews on the live training DiT at the template's settings until the turbo path is measured
    # on Fizgig (Krea 2 pattern: live training model, family turbo LoRA, no model swap).
    preview_steps=25,
    preview_cfg=1.0,
    preview_width=1024,
    preview_height=1024,

    presets=(
        # The recipe measured on 26 Sep 2026 (170-image face set): portrait likeness 76-77 from epoch 6 on, best
        # around step 2700 (~16 epochs at 170 images). Rank 32, Fizgig adapter on, Adaptive LR 1e-4..2e-4.
        ("✨ Qwen 2.1 Standard (rank 32, training adapter, adaptive LR)", {
            "NETWORK_DIM": 32, "NETWORK_ALPHA": 32, "LEARNING_RATE": 1e-4,
            "MAX_TRAIN_EPOCHS": 16, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
            "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "1e-4", "ADAPTIVE_LR_MAX": "2e-4",
            "OPTIMIZER_TYPE": "adamw", "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0,
            "DATASET_MEGAPIXELS": "1.0", "FAMILY_TRAINING_ADAPTER": True,
        }),
    ),
    workbench={},                     # no workbench tab yet: training first (Peter, #155)

    notes=(
        ("A plain LoRA is unstable: collapse to texture at lr 5e-4 (~step 300), wobble at 1e-4, a no-adapter "
         "Adaptive-LR run fell to 37 likeness at step 2000. Loss does not show it. Train with the frozen "
         "training adapter: 77.4 likeness (Fizgig adapter) vs 76.4 (SimpleTuner v2) vs 55.8 (none), and far "
         "better detail with Fizgig's (Peter's eye). Adaptive LR 1e-4..2e-4; ~2750 steps on a 170-image set.",
         "Fizgig lab A/B 26 Sep 2026 (Desktop qwen_ab_results); SimpleTuner assistant-v2 card"),
        ("Text encoder conditioning is the LAST layer BEFORE the final RMSNorm, with a fixed system-prompt "
         "template whose tokens are dropped. transformers >= 5 returns the normed state unless hooked.",
         "diffusers pipeline_qwenimage21.py L206-310"),
        ("Block-causal attention (text causal, each image bidirectional) and t=0 modulation for the text "
         "prefix (causal_condition) are required in every training forward.",
         "diffusers transformer_qwenimage21.py L238-306"),
        ("Gradient checkpointing with only block Linears LoRA-wrapped records no graph unless the joint "
         "hidden states require grad; layer offload produced NaN grads in ai-toolkit.",
         "ostris/ai-toolkit#1054"),
    ),
)
