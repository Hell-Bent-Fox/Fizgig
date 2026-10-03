"""MiniMax H3 through the standard layer, beside the old H3 trainer ("MiniMax H3 (driver)"), hidden from the Base Model
list until it trains end to end. Same model files and Preferences rows as MiniMax H3 (shares_prefs_with), the same
arch id and cache layout (minimaxh3 - the dataset layer's clip / voice discovery keys on it, and the 32B text caches
are reused), kohya LoRA keys as the old trainer writes them. Facts are the old trainer's (src/fizgig/minimax/), cited
per value.
"""
from fizgig.families.description import ClipSpec, FamilyDescription, LoRAFormat, ModelFile, SamplingSettings

MINIMAX = FamilyDescription(
    key="minimax_driver",
    arch_id="minimaxh3",
    display_name="MiniMax H3",
    gui_label="MiniMax H3 (driver)",
    lora_name_suffix="mmh3",
    experimental=True,
    hidden=True,                      # not offered in the GUI until the driver trains and previews

    # the MiniMax H3 Preferences rows (lora_trainer_gui.py DEFAULT_PREFS minimax_*); download links stay on that card
    model_files=(
        ModelFile("minimax_dit", "DiT (fl2va, int8 ConvRot)", True, role="dit",
                  note="The pruned int8 ConvRot base the LoRA trains on."),
        ModelFile("minimax_vae", "Video VAE", True, role="vae"),
        ModelFile("minimax_text_encoder", "Qwen3-VL-32B text encoder", True, role="text_encoder",
                  note="Used for caching only (nvfp4), then unloaded before training steps."),
        ModelFile("minimax_audio_vae", "Audio VAE", False, role="audio_vae",
                  note="Clips' sound and voice recordings; without it clips train their pictures only."),
        ModelFile("minimax_turbo_lora", "Turbo LoRA (previews)", False, role="speed_lora"),
        ModelFile("minimax_ref_dit", "DiT (ref2va, optional)", False, role="ref_dit"),
        ModelFile("minimax_circlestone_adapter", "Circlestone adapter (photos)", False, role="training_adapter"),
        ModelFile("minimax_training_adapter", "Ostris adapter (fl2va)", False),
        ModelFile("minimax_ref_training_adapter", "Ostris adapter (ref2va)", False),
    ),
    shares_prefs_with="minimax",
    text_encoder_label="Qwen3-VL-32B",
    vae_label="MiniMax H3 Video VAE",

    media=("photo", "clip", "voice"),
    clip_spec=ClipSpec(fps=24, frame_step=17, frame_offset=5, edge_multiple=32, audio_rate=32000, audio_channels=2,
                       mute_suffix="_mute", note="src/fizgig/minimax/clip.py FPS / GRID_FRAMES / SIZE_STEP"),

    latent_channels=24,
    spatial_factor=16,                # dataset/image_dataset.py LATENT_SPATIAL_FACTOR[minimaxh3]
    bucket_step=32,                   # BUCKET_RESO_STEPS[minimaxh3]: 16x VAE x 2x2 patch
    image_channels=3,
    native_megapixels=0.6,            # 768 short edge (the old Samples default)

    n_blocks=50,                      # MiniMaxH3Config.num_layers
    block_prefix="blocks",
    block_note="Block ids follow the old Repair Studio (h3blk_0-49, h3_rf_0-1), so its saved presets apply.",

    lora=LoRAFormat(
        key_template="lora_unet_blocks_{block}_{module}.{ab}.weight",
        down="lora_down", up="lora_up",
        block_modules=("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"),
        alpha_key="{prefix}.alpha",
        kohya=True,
        file_prefix="lora_unet_",
        note="kohya keys as the old H3 trainer writes them (create_network(None, 'lora_unet', ...)); the AdaLN "
             "projection is left out by default (--no_train_adaln).",
        source="src/fizgig/minimax/trainer.py train_minimax",
    ),

    driver="fizgig.minimax.driver:MiniMaxDriver",
    modelspec_arch="MiniMax-H3",
    implementation="https://github.com/MiniMax-AI/MiniMax-H3",
    ema_default="0.98",               # the old H3 default (v5.4.1)
    precisions=("int8",),             # the native int8 ConvRot base (quant tiers + rings come with the H3 port)
    optimizers=("automagic3", "adamw8bit", "adamw"),
    network_types=("lora", "lokr"),

    sampling=(
        SamplingSettings("H3 reference", steps=20, cfg=1.0, sampler="euler", scheduler="simple",
                         note="CFG-free on the fixed shift-12 schedule, as the shipped ComfyUI workflows.",
                         source="lora_trainer_gui.py ARCHITECTURES['MiniMax H3'] sample defaults"),
    ),
    preview_steps=20,
    preview_cfg=1.0,
    preview_width=768,
    preview_height=768,
)
