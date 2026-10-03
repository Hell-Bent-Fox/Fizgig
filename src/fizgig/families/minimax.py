"""MiniMax H3 through the standard layer, beside the old H3 trainer ("MiniMax H3 (driver)"), hidden from the Base Model
list until it trains end to end. Same model files and Preferences rows as MiniMax H3 (shares_prefs_with), the same
arch id and cache layout (minimaxh3 - the dataset layer's clip / voice discovery keys on it, and the 32B text caches
are reused), kohya LoRA keys as the old trainer writes them. Facts are the old trainer's (src/fizgig/minimax/), cited
per value.
"""
from fizgig.families.description import (ClipSpec, FamilyDescription, FamilyOption, LoRAFormat, ModelFile,
                                        SamplingSettings, SpeedLoRA)

# The old H3 Training-tab controls as family options (lora_trainer_gui.py's MiniMax rows and _build_minimax_command),
# their old settings keys kept so presets, queued runs and Last Train carry across.
_OSTRIS = "pref:minimax_training_adapter[H3_TRAIN_BASE=ref2va|H3_DISTILL=1->minimax_ref_training_adapter]"
# Training Structure: the old dropdown, a view of the clean-end share (MINIMAX_LOWNOISE_PCT); Custom reveals the box
_STRUCTURE = (
    ("Likeness and Style — 60% clean-end", "lownoise_pct=60", "60",
     "Most of the run on nearly-clean images — the tuned default for stills. See the MiniMax section of the README."),
    ("Model default, movement — 8% clean-end", "lownoise_pct=8", "8",
     "The reference trainer's schedule, weighted to movement and composition. See the MiniMax section of the README."),
    ("Custom", "", "", "Type your own clean-end share. See the MiniMax section of the README."),
)


def _preset(rank, epochs, clip_still=True):
    """The old H3 built-ins (lora_trainer_gui.py MINIMAX_BUILT_IN_PRESETS), values unchanged, in the standard layer's
    keys and this family's option keys."""
    return {
        "NETWORK_DIM": rank, "NETWORK_ALPHA": rank, "NETWORK_TYPE": "LoRA (standard)", "LOKR_FACTOR": 8,
        "LEARNING_RATE": 1e-6, "MAX_TRAIN_EPOCHS": epochs, "SAVE_EVERY_N_EPOCHS": 1, "SEED": 42,
        "ADAPTIVE_LR": False, "ADAPTIVE_LR_MIN": "1e-5", "ADAPTIVE_LR_MAX": "4e-4", "OPTIMIZER_TYPE": "automagic3",
        "GRADIENT_ACCUMULATION": 1, "MAX_GRAD_NORM": 1.0, "DATASET_MEGAPIXELS": "0.25",
        "FAMILY_PRECISION": "Auto (recommended)", "BLOCKS_SWAP": "Auto (detect from GPU)",
        "FAMILY_EMA": "0.98 (recommended)",
        "H3_ADAPTER_RAMP": "Off", "H3_CAPTION_DROPOUT": "0.05 (default)", "H3_STRUCTURE": _STRUCTURE[0][0],
        "H3_LOWNOISE_PCT": "60", "H3_HIGHNOISE_LR_PCT": "100",
        "H3_BLOCKS": "all", "H3_TRAIN_REFINER": "", "H3_LIKENESS_MODE": "Default",
        "H3_ADAPTER": "Circlestone — best for photos", "H3_TREAD": "1", "H3_CLIP_STILL": "1" if clip_still else "",
        "H3_DISTILL": "",
    }


PRESETS = (
    ("✨ MiniMax H3 Fast (LoRA 8, 50 epochs)", _preset(8, 50)),
    ("✨ MiniMax H3 (rank 16, 60 epochs)", _preset(16, 60)),
    ("✨ MiniMax H3 Style (LoRA 8)", _preset(8, 50, clip_still=False)),
)

OPTIONS = (
    FamilyOption("H3_TRAIN_BASE", "Training Base", tab="model", choices=(
        ("First/last frame (fl2va) — standard", ""),
        ("Reference (ref2va)", "--dit=pref:minimax_ref_dit")),
        hint="Reference (ref2va) if the LoRA lives in the r2v workflow; needs DiT (reference) in Preferences.",
        setting="MINIMAX_TRAIN_BASE"),
    FamilyOption("H3_STRUCTURE", "Training Structure", choices=tuple((c[0], c[1]) for c in _STRUCTURE),
                 choice_values=tuple(c[2] for c in _STRUCTURE), choice_notes=tuple((c[0], c[3]) for c in _STRUCTURE),
                 setting="MINIMAX_LOWNOISE_PCT"),
    FamilyOption("H3_LOWNOISE_PCT", "Clean-end share (%)", kind="entry", tokens="lownoise_pct={}", default="60",
                 hint="The share of steps drawn at the clean (detail) end: 60 is Likeness and Style, 8 the model's "
                      "default.", setting="MINIMAX_LOWNOISE_PCT", requires="H3_STRUCTURE=Custom"),
    FamilyOption("H3_HIGHNOISE_LR_PCT", "Medium to High Noise LR", kind="entry",
                 tokens="highnoise_lr_pct={}", default="100",
                 hint="Scales the LR of the noisy-half steps: pose, framing, face shape. Leave at 100 unless "
                      "experimenting.",
                 setting="MINIMAX_HIGHNOISE_LR_PCT"),
    FamilyOption("H3_MIXED_STOP_CATEGORY", "Finish one category early", choices=(
        ("voice", "stop_category=audio"), ("photos & clips", "stop_category=visual")),
        setting="MIXED_STOP_CATEGORY", mixed_only=True),
    FamilyOption("H3_MIXED_STOP_EPOCH", "After epoch", kind="entry", tokens="stop_epoch={}",
                 setting="MIXED_STOP_EPOCH", mixed_only=True),
    FamilyOption("H3_MIXED_STOP_MODE", "Then", choices=(
        ("anchor at 10% LR (recommended)", "stop_mode=anchor"), ("stop completely (faster)", "stop_mode=stop")),
        hint="Finish the smaller category early, before it overbakes. Blank = both train to the end. Anchor holds it "
             "at 10% LR and keeps its epoch report live. Stop skips its steps: faster, but unwatched.",
        setting="MIXED_STOP_MODE", mixed_only=True),
    FamilyOption("H3_LIKENESS_MODE", "Training mode", choices=(
        ("Default", "photo_blocks=20-49 clip_blocks=20-49 audio_blocks=20-49"),
        ("More Blocks", "--train_blocks=6-49"),
        ("Off · hand-pick the blocks below", "")),
        choice_hints=(
            ("Default", "High quality, versatile, best at preserving model priors. Photos, clips and voice all train "
                        "blocks 20-49, and the backward stops at the window so the steps are quicker too."),
            ("More Blocks", "Less preservation of model priors, high quality. May help when you are training a MOTION "
                            "concept specifically, since it reaches more of the model. It is not a likeness upgrade - "
                            "Default reaches higher likeness, sooner, with quicker steps. Every step type trains 6-49, "
                            "at 44 blocks in the backward instead of 30. Blocks 0-5 stay out either way; they deform "
                            "anatomy and colour."),
            ("Off · hand-pick the blocks below", "The blocks are yours to pick, for experiments: Blocks to Train.")),
        setting="MINIMAX_LIKENESS_MODE"),
    FamilyOption("H3_ADAPTER", "Training adapter", choices=(
        ("Circlestone — best for photos", "--training_adapter=pref:minimax_circlestone_adapter"),
        ("Ostris — best for videos", f"--training_adapter={_OSTRIS}"),
        ("Off", "")),
        hint="De-distills the base while your LoRA learns: frozen at 1.0 for every training step, off for previews "
             "and never in your saved file. Circlestone (one file for fl2va and ref2va) trains sharper LoRAs from "
             "photos; Ostris learns a video look faster.",
        setting="MINIMAX_ADAPTER"),
    FamilyOption("H3_TREAD", "TREAD token routing — on clip steps, half the video tokens skip the middle blocks",
                 kind="check", tokens="tread=0.5@2-47", default="1",
                 hint="Faster clip steps: a random half of each clip's video tokens skips blocks 2-46 and rejoins "
                      "unchanged. Photos and clip stills always run in full; previews and your saved LoRA are "
                      "untouched. See the MiniMax section of the README.",
                 mode="lora", setting="MINIMAX_TREAD", show_if_media="clip"),
    FamilyOption("H3_CLIP_STILL", "Also train each clip's sharpest face frame as a photo", kind="check",
                 tokens="clip_still_as_photo=1 aux:clip_still=1", default="1",
                 hint="Each clip's sharpest face frame trains as a photo with the clip's caption. Picked at caching; "
                      "clips cached with this off use frame 0 until re-cached.",
                 setting="MINIMAX_CLIP_STILL", show_if_media="clip"),
    FamilyOption("H3_ADAPTER_RAMP", "Adapter-relative LR", kind="choice", choices=(
        ("Off", ""), ("0.003 (slow build)", "adapter_ramp=0.003"), ("0.005 (recommended)", "adapter_ramp=0.005"),
        ("0.01 (fast build)", "adapter_ramp=0.01")),
        hint="Makes the Learning Rate box a ceiling the run climbs toward. Set it where you want to end up.",
        setting="MINIMAX_ADAPTER_RAMP", section="other"),
    FamilyOption("H3_CAPTION_DROPOUT", "Caption dropout", choices=(
        ("Off", "caption_dropout=0"), ("0.05 (default)", "caption_dropout=0.05"),
        ("0.10 (strong)", "caption_dropout=0.1")), choice_values=("0", "0.05", "0.1"), default="0.05 (default)",
        hint="Trains a few percent of steps with no caption, so the LoRA does not lean entirely on the trigger word.",
        setting="MINIMAX_CAPTION_DROPOUT", section="other"),
    FamilyOption("H3_BLOCKS", "Blocks to Train", kind="entry", tokens="--train_blocks={}", default="all",
                 suggestions=("6-49 · recommended (skips 0-5)", "all · every block (50 of 50)",
                              "10-49 · skip the first 10", "14-37 · middle band", "25-49 · back half",
                              "0-24 · front half"),
                 hint="Train a subset of the 50 blocks. Type ranges and singles, comma-separated, like 3-12, 22, "
                      "31-33. Measured answers: 6-49 for the whole model (what More Blocks runs) and 20-49 for "
                      "likeness (Default). Blocks 0-5 are in neither: they deform anatomy and pull the dataset's "
                      "colour into the render.",
                 setting="MINIMAX_BLOCKS", requires="H3_LIKENESS_MODE=Off", section="other"),
    FamilyOption("H3_DISTILL", "Learn identity from my dataset (reference distillation)", kind="check",
                 tokens="distill=1 aux:distill=1 --dit=pref:minimax_ref_dit",
                 hint="Experiment. Teaches the LoRA to reproduce identity the way H3 does from a reference photo. "
                      "Needs the ref2va model in Preferences.", setting="MINIMAX_DISTILL", section="other"),
    FamilyOption("H3_DISTILL_REFS", "References per photo", choices=(
        ("2", "aux:distill_refs=2"), ("1", "aux:distill_refs=1"), ("3", "aux:distill_refs=3"),
        ("4", "aux:distill_refs=4")), requires="H3_DISTILL", setting="MINIMAX_DISTILL_REFS", section="other"),
    FamilyOption("H3_DISTILL_WEIGHT", "Teacher weight", kind="entry", tokens="distill_weight={}", default="0.8",
                 requires="H3_DISTILL", setting="MINIMAX_DISTILL_WEIGHT", section="other"),
    FamilyOption("H3_DISTILL_PHASE1", "Identity-first phase", choices=(
        ("Auto (from dataset size)", "distill_phase1=-1"), ("Off — blend throughout", "distill_phase1=0"),
        ("2 epochs", "distill_phase1=2"), ("4 epochs", "distill_phase1=4"), ("8 epochs", "distill_phase1=8"),
        ("16 epochs", "distill_phase1=16"), ("30 epochs", "distill_phase1=30")), requires="H3_DISTILL", setting="MINIMAX_DISTILL_PHASE1", section="other"),
    FamilyOption("H3_TRAIN_REFINER", "Train the text token refiner", kind="check", tokens="train_token_refiner=1",
                 hint="Recommended off. Does not affect the ability to use a trigger word. The refiner sets how every "
                      "prompt is read; training it softens output and makes previews judder between epochs. LoRA and "
                      "fine-tune runs alike (under fine-tune it would train alongside every window, four times the "
                      "duty cycle of any block).",
                 setting="MINIMAX_TRAIN_REFINER", section="other"),
    FamilyOption("H3_SAMPLE_FRAMES", "Sample length", tab="samples", choices=(
        ("Still (1 frame)", "preview_frames=1"),
        ("22 frames with sound (~1s)", "preview_frames=22 preview_audio=1"),
        ("56 frames with sound (~2.3s)", "preview_frames=56 preview_audio=1"),
        ("124 frames with sound (~5s)", "preview_frames=124 preview_audio=1")), setting="SAMPLE_FRAMES"),
    FamilyOption("H3_FT_SCOPE", "Train on", choices=(("All media", ""), ("Photos only", "ft_scope=photo")),
                 hint="A dataset FILTER, not a mode. All media fine-tunes on everything in the folder - photos, clips, "
                      "voice. Photos only skips the clips and voice of a mixed folder (with Training mode on Default "
                      "the cycle then tightens to the identity blocks).",
                 setting="MINIMAX_FT_SCOPE", mode="finetune"),
    FamilyOption("H3_FT_BLOCKS", "Fine-tune blocks", kind="entry", tokens="ft_blocks={}",
                 hint="Optional: restrict the rotation cycle to a block range - the whole fine-tune touches only these "
                      "blocks. 20-49 is the measured likeness recipe (protects the fragile 0-19 trunk) and roughly "
                      "halves the system-RAM master copy. Empty = the full model.",
                 setting="MINIMAX_FT_BLOCKSPEC", mode="finetune"),
    FamilyOption("AUDIO_VAE", "", kind="fixed", tokens="audio_vae=pref:minimax_audio_vae aux:audio_vae=pref:minimax_audio_vae"),
)

MINIMAX = FamilyDescription(
    key="minimax_driver",
    arch_id="minimaxh3",
    display_name="MiniMax H3",
    gui_label="MiniMax H3 (driver)",
    lora_name_suffix="mmh3",
    experimental=True,
    hidden=False,

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

    options=OPTIONS,
    presets=PRESETS,
    settings_aliases={"FAMILY_EMA": "MINIMAX_EMA", "FAMILY_PRECISION": "MINIMAX_BASE_QUANT", "FAMILY_FT": "MINIMAX_FINETUNE",
                      "FAMILY_FT_ROTATE_EVERY": "MINIMAX_FT_EVERY", "FAMILY_FT_FUSED": "MINIMAX_FT_FUSED",
                      "FAMILY_FT_REG_DIR": "MINIMAX_REG_DIR", "FAMILY_FT_REG_MULT": "MINIMAX_REG_MULT"},
    workbench=("repair", "explorer", "profiler", "extract", "royale"),
    workbench_engine="fizgig.minimax.workbench:H3WorkbenchEngine",
    finetune=True,
    ft_learning_rate=3e-5,            # the tested H3 fine-tune rate (1e-4 destroys; 1e-5 too slow to judge from)                    # the old rotation FT (component windows on an NF4 trunk, int8 ConvRot saves)
    media=("photo", "clip", "voice"),
    multi_concept=True,
    adaptive_lr=False,
    loss_watch=False,
    network_hint="LoRA recommended for MiniMax",
    ema_hint="A smoothed average of the weights, leading to better and more reliable previews.",
    samples_turbo_pace=True,
    preview_park_optimizer=True,      # the old previews' optimizer-state park (~2.5 GB back for the render)
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
    ema_short_run=True,
    resumes_untagged_states=True,     # an old H3 pause: same files, same parameter order (blocks, qkv/out/fc1/fc2)
    # the training adapter is the MINIMAX_ADAPTER option (Circlestone / Ostris per base / Off), not the generic tick
    # the checkpoint's own int8 ConvRot codes, or a 4-bit base (NF4, HQQ); Auto is the driver's plan_run (the old
    # planner), which streams blocks H2D rather than giving up the int8 base
    precisions=("int8", "nf4", "hqq"),
    precision_labels={"auto": "Auto (recommended)", "int8": "int8 · most accurate, needs ~30 GB free",
                      "nf4": "4-bit · fits smaller cards", "hqq": "4-bit HQQ · lower error than 4-bit, slower"},
    precision_hint="Auto reads free VRAM at launch and picks precision and block swap together. int8 is the most "
                   "accurate, 4-bit fits smaller cards, 4-bit HQQ sits between (less error, more VRAM, slower with no "
                   "swap). Auto never picks HQQ.",
    optimizers=("automagic3", "adamw8bit", "adamw"),
    optimizer_weight_decay=1e-4,      # the old trainer's (ai-toolkit's job template); bnb's default is 1e-2
    optimizer_eps_floor_8bit=True,
    trainable_dtype="bf16",           # the old trainer: network.to(device, dtype=bfloat16)
    network_types=("lora", "lokr"),

    sampling=(
        SamplingSettings("H3 reference", steps=20, cfg=1.0, sampler="euler", scheduler="simple",
                         note="CFG-free on the fixed shift-12 schedule, as the shipped ComfyUI workflows.",
                         source="lora_trainer_gui.py ARCHITECTURES['MiniMax H3'] sample defaults"),
    ),
    speed_loras=(
        SpeedLoRA(
            name="H3 Turbo LoRA (6-step)",
            repo="larryvrh/MiniMax-H3-Turbo-Lora",
            file="minimax_h3_turbo_v4_step600.safetensors",
            pairs_with="MiniMax H3 fl2va",
            strength=0.75,
            settings=SamplingSettings("Turbo 6-step", steps=6, cfg=1.0, sampler="euler", scheduler="simple",
                                      note="CFG-free; the old previews' 6 steps at 0.75.",
                                      source="src/fizgig/minimax/trainer.py load_preview_turbo"),
            load_unmerged=True,
            pref_key="minimax_turbo_lora",
            source="https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora",
        ),
    ),
    preview_speed_lora="H3 Turbo LoRA (6-step)",
    preview_steps=20,
    preview_cfg=1.0,
    preview_width=768,
    preview_height=768,
)
