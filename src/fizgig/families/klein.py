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


def _preset(rank, lr, epochs, area, adaptive, ts=("", "")):
    """The old Klein built-ins (lora_trainer_gui.py BUILT_IN_PRESETS), values unchanged, in the standard layer's
    keys. adaptive = (min, max)."""
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LEARNING_RATE": lr,
        "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": adaptive[0], "ADAPTIVE_LR_MAX": adaptive[1],
        "TARGET_LAYERS": area, "MIN_TIMESTEP": ts[0], "MAX_TIMESTEP": ts[1], "OPTIMIZER_TYPE": "adamw8bit",
        "FAMILY_PRECISION": "Auto (fits your free VRAM)", "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_EMA": "Off",
    }


_STYLE_COMP = tuple(f"double_{i}" for i in range(8)) + ("single_0", "single_1")

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
    # the old Repair Studio's 5 buckets (lora_trainer_gui.py _repair_category_for_block): style+composition in the
    # double blocks and single 0, the style/identity overlap at single 1, identity 2-11, the identity/detail overlap
    # 12-16, details 17-23
    block_categories=tuple((f"double_{i}", "style_composition") for i in range(8)) + tuple(
        (f"single_{i}", "style_composition" if i == 0 else "style_ident_overlap" if i == 1 else
         "identity" if i <= 11 else "ident_details_overlap" if i <= 16 else "details") for i in range(24)),
    category_masters=True,
    workbench=("repair", "explorer", "profiler", "extract", "royale"),

    # Model Area to Train, the old Training tab's areas and their block patterns (lora_trainer_gui.py, Klein's
    # include_patterns): Style trains the style+composition blocks at the late (clean) timesteps 0-400
    train_areas=(
        ("Full Model", (), None),
        ("Identity", tuple(f"single_{i}" for i in range(1, 17)), None),
        ("Style", _STYLE_COMP, (0.0, 0.4)),
        ("Style+Composition", _STYLE_COMP, None),
        ("Details", tuple(f"single_{i}" for i in range(12, 24)), None),
    ),

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
    precision_labels={"bf16": "As the file (bf16 or fp8)"},
    precision_hint=("Auto (recommended) picks at launch: INT8 compiled when torch.compile can run and the run is long "
                    "enough (the fastest Klein setup), otherwise your Base file as it is - BFL's fp8 file trains in "
                    "fp8, as Klein always has. Too little free VRAM for those: 4-bit NF4. \"As the file\" never "
                    "requantises; INT8 and NF4 requantise from the file."),
    # Measured 3 Oct 2026 on a 5090, full model, rank 32, adamw8bit, gradient checkpointing, BFL's fp8 base file, peak
    # reserved over the first epoch (24 photos): INT8 12.6 GB at 0.25 MP / 17.1 GB at 1 MP (1.18 / 2.51 s/step); NF4 7.9 /
    # 9.9 GB (1.06 / 2.02 s/step). Swapped-block savings not measured yet (0 = Auto does not plan a swap)
    train_memory={"int8": (((0.25, 12.6), (1.0, 17.1)), 0.0), "nf4": (((0.25, 7.9), (1.0, 9.9)), 0.0)},
    optimizers=("adamw8bit", "adamw"),
    network_types=("lora", "lokr"),
    adaptive_lr_clip_signal=True,     # the old trainer's grad-clip ratio signal (K6)
    # torch.compile of both block lists (the driver's compile_blocks); graph breaks tolerated, as the old trainer's
    # default. Measured 3 Oct 2026 on a 5090, full model, rank 32, 0.25 MP (epoch-3 s/step, peak GB):
    #   INT8  eager 1.14 / 12.8   inside 0.47 / 20.7   outside 0.54 / 11.8   (1 MP: eager 2.49 / 17.8, outside 1.29 / 15.7)
    #   NF4   eager 0.96 / 7.9    inside 0.67 / 7.3                          (1 MP: eager 2.02 / 9.9, inside 1.49 / 9.3)
    #   fp8 file (bf16 choice)  eager 0.77 / 11.1   inside 0.69 / 11.9   outside fails (the driver compiles it inside)
    # First epoch while compiling: INT8 outside 4.2 s/step, NF4 inside 5.5 -> pays back after ~130 / ~380 steps.
    compiles=True,
    compile_fullgraph=False,
    compile_boundary="outside",
    compile_payback_steps={"int8": 200, "nf4": 400},
    compile_memory={"int8": {"inside": ((0.25, 20.7),), "outside": ((0.25, 11.8), (1.0, 15.7))},
                    "nf4": {"inside": ((0.25, 7.3), (1.0, 9.3))}},
    compile_hint=("Auto (recommended) turns torch.compile on only when this run is long enough to repay it. On Klein "
                  "9B it is measured 2.1x per step on the INT8 base (1.14 -> 0.54 s/step) with LESS memory than "
                  "uncompiled (the gradient checkpoint stays outside the compiled blocks), and 1.4x on NF4. The first "
                  "epoch runs slower while the blocks compile, so Auto waits for runs longer than about 200 steps on "
                  "INT8 and 400 on NF4, and checks the compiled run fits your free VRAM. The fp8 base is not "
                  "compiled by Auto (1.1x; On still compiles). Requires Triton and, on Windows, a C++ compiler (VS "
                  "Build Tools) - both located automatically. Never used with Blocks Swap, since swapping moves "
                  "weights and compiled graphs assume they stay put."),
    edit_training=True,               # Klein is an edit model: references ride after the image tokens
    edit_note=("Pairs of an original and its edited version, the same crop and shape - about 40 pairs is a good "
               "start (20 at least). Each photo 1 MP or larger; Fizgig resizes them."),
    slider_training=True,
    train_preview_checkpoint=True,    # the old "Use Distilled model for samples" (on by default)
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
    preview_steps=40,                 # the old Klein entry's Base sample defaults: 40 steps, CFG 4.5, 768x768
    preview_cfg=4.5,
    preview_cfg_note="Base has no guidance embed, so CFG is the only guidance: 1 = none (the negative prompt is "
                     "ignored); about 3.5 to 4.5 gives guided previews.",
    preview_width=768,
    preview_height=768,
    presets=(
        ("✨ Old Reliable (rank 16, full model, single subject)", _preset(16, 1e-4, 55, "Full Model", ("1e-4", "4e-4"))),
        ("✨ Old Reliable - Flavour 8 (rank 8, full model, single subject)",
         _preset(8, 1e-4, 55, "Full Model", ("1e-4", "4e-4"))),
        ("✨ Identity (rank 8, single subject)", _preset(8, 4e-4, 15, "Identity", ("2e-4", "4e-4"))),
        ("✨ Identity (rank 8, harder dataset)", _preset(8, 4e-4, 20, "Identity", ("2e-4", "4e-4"))),
        ("✨ Multi-Character (rank 16, multi character or concept)",
         _preset(16, 2e-4, 50, "Identity", ("1e-4", "4e-4"))),
        ("✨ Style (late timesteps)", _preset(4, 4e-4, 15, "Style", ("1e-5", "4e-4"), ("0", "400"))),
        ("✨ Style+Composition (all timesteps)", _preset(4, 4e-4, 15, "Style+Composition", ("1e-5", "4e-4"))),
    ),
    notes=(
        ("Timesteps: flux2_shift - logit-normal, shift exp(mu) with mu 0.5 at 256 latent tokens to 1.15 at 4096, "
         "unweighted MSE on the flow-matching velocity.",
         "src/fizgig/training/trainer.py get_noisy_model_input_and_timesteps"),
    ),
)
