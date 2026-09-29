"""FamilyDescription: everything Fizgig needs to know about one model family, in one object.

Adding a family used to mean threading it through the GUI by hand (predicates, if-chains, Preferences
blocks, preset dicts, command builders). A description holds the family's facts once; the GUI's generic
paths read it. Klein, Krea 2 and MiniMax H3 are NOT described here: they keep their existing code paths
untouched until this system has proven itself on a new family (Peter, 25 Sep 2026).

Every value that came from outside Fizgig carries its source (`source=` fields), so a later reader can
re-check it when the upstream model or a speed LoRA moves on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelFile:
    """One file the user points Fizgig at in Preferences."""
    pref_key: str                     # prefs.json key, e.g. "qwen21_dit"
    label: str                        # Preferences row label
    required: bool = True             # training cannot start without it
    repo: str = ""                    # Hugging Face repo for the Download link
    path: str = ""                    # file path inside the repo
    size_gb: float = 0.0
    note: str = ""                    # one plain line shown under the row
    local_name: str = ""              # name in models/ when the repo's own is generic (diffusion_pytorch_model...)
    role: str = ""                    # "dit" | "vae" | "text_encoder" | "training_adapter" | "speed_lora" | ""


@dataclass(frozen=True)
class SamplingSettings:
    """A complete, citable way to sample the family: steps, CFG, sampler, schedule."""
    name: str
    steps: int
    cfg: float
    sampler: str = "euler"
    scheduler: str = "simple"
    sigmas: Optional[tuple] = None    # explicit schedule when the model needs one
    options: tuple = ()               # driver-specific sampler options as (name, value) pairs
    negative_prompt: bool = False     # whether a negative prompt does anything at this CFG
    note: str = ""
    source: str = ""


@dataclass(frozen=True)
class SpeedLoRA:
    """A Turbo / Lightning / distill LoRA for the family, with the settings it actually wants."""
    name: str
    repo: str
    file: str
    pairs_with: str                   # which base it was trained against
    strength: float
    settings: SamplingSettings
    load_unmerged: bool = False       # merging into the weights loses part of it
    pref_key: str = ""                # the model file (Preferences row) holding it
    community_settings: tuple = ()    # (description, source) pairs: how people actually use it
    caveats: tuple = ()
    source: str = ""


@dataclass(frozen=True)
class LoRAFormat:
    """How this family's LoRA keys are written so ComfyUI (and diffusers) load every module."""
    key_template: str                 # e.g. "transformer.transformer_blocks.{block}.{module}.{ab}.weight"
    down: str                         # "lora_A" / "lora_down"
    up: str                           # "lora_B" / "lora_up"
    block_modules: tuple              # per-block Linears the LoRA targets
    alpha_key: str = "{prefix}.alpha"
    kohya: bool = False               # True = the lora_unet_ convention used by Klein/Krea 2/H3
    file_prefix: str = ""             # what precedes a module path in every key, e.g. "transformer."
    note: str = ""
    source: str = ""

    def module_of(self, key: str) -> Optional[str]:
        """Dotted module path of a down-weight key ('transformer.modulation.1.lora_A.weight' -> 'modulation.1'),
        None for any other key."""
        tail = f".{self.down}.weight"
        if not (key.startswith(self.file_prefix) and key.endswith(tail)):
            return None
        return key[len(self.file_prefix):-len(tail)]

    def key(self, block: int, module: str, which: str) -> str:
        """which: "down" or "up"."""
        ab = self.down if which == "down" else self.up
        return self.key_template.format(block=block, module=module, ab=ab)


@dataclass(frozen=True)
class FamilyDescription:
    # identity
    key: str                          # family key (the workbench vocabulary), e.g. "qwen_image21"
    arch_id: str                      # architecture id used in cache filenames, e.g. "qwenimage21"
    display_name: str                 # "Qwen Image 2.1"
    gui_label: str                    # Base Model selector entry
    lora_name_suffix: str
    aliases: tuple = ()
    experimental: bool = True

    # model files (Preferences rows, in display order)
    model_files: tuple = ()
    text_encoder_label: str = ""
    vae_label: str = ""

    # latent rules
    latent_channels: int = 16
    spatial_factor: int = 8
    bucket_step: int = 64             # training buckets snap to this many pixels
    image_channels: int = 3           # 4 = RGBA
    native_megapixels: float = 1.0

    # block layout (Repair Studio / block targeting)
    n_blocks: int = 0
    block_prefix: str = ""            # "transformer_blocks"
    block_note: str = ""

    # LoRA
    lora: Optional[LoRAFormat] = None

    # the family's driver ("module.path:ClassName", a FamilyDriver); empty = not built yet, so the family stays
    # hidden from training. Caching and training run through the generic entry points below for every family.
    driver: str = ""
    modelspec_arch: str = ""          # SAI modelspec.architecture, e.g. "Qwen-Image-2.1"
    training_adapter: str = ""        # pref key of the family's frozen training adapter ("" = none)
    training_adapter_note: str = ""   # one line for the Training tab under the adapter toggle
    ema_default: str = ""             # default EMA decay for the Training tab ("0.98", "Off"); "" = no EMA control
    implementation: str = ""          # SAI modelspec.implementation (reference repo URL)
    precisions: tuple = ("bf16",)     # base precisions offered for training: any of "bf16", "int8", "nf4"
    # measured training memory for the Auto plan: {precision: (peak GB with no block swap, GB saved per swapped
    # block)}; the peak may instead be ((megapixels, GB), ...) points, interpolated for the run's resolution.
    # {} = Auto just takes the first precision
    train_memory: dict = field(default_factory=dict)
    optimizers: tuple = ("adamw8bit", "adamw")
    network_types: tuple = ("lora",)
    edit_training: bool = False       # Edit LoRA from before/after pairs (the driver's supports_references)
    edit_note: str = ""               # the Edit LoRA section's "What you need" line: pair count and photo size
    slider_training: bool = False     # Slider LoRAs (strength is a dial between two looks); the driver needs
    #                                   training_loss(diff_ref=, diff_weight=) and noise_latents / predict

    # sampling
    sampling: tuple = ()              # SamplingSettings without any speed LoRA (first = default)
    speed_loras: tuple = ()           # SpeedLoRA entries
    preview_steps: int = 20
    preview_cfg: float = 1.0
    preview_width: int = 1024
    preview_height: int = 1024
    preview_speed_lora: str = ""      # name of the SpeedLoRA previews use when its file is set in Preferences
    preview_speed_steps: int = 0      # preview steps with it (0 = the SpeedLoRA's own)
    preview_speed_strength: Optional[float] = None   # preview strength for it (None = the SpeedLoRA's own; 0 = off
    # by default: previews render without it until the Samples tab's Turbo strength is raised)
    # (steps, strength) preview defaults an older release shipped; the Samples tab replaces them with the current
    # ones, so a setting saved under the old default moves on instead of sticking
    retired_preview_defaults: tuple = ()
    # a one-time reset: when this tag is new to a user, the Samples tab replaces their saved preview steps and turbo
    # strength with the current defaults once (the tag is remembered, so later choices stick). Change it to reset again.
    preview_reset: str = ""

    # built-in Training-tab presets: ((name, {GUI setting key: value}), ...); the first is applied on a first visit
    presets: tuple = ()

    # small files the text encoder / captioner load by repo name: ((repo, (allow_patterns...)), ...); the model
    # downloader fetches them with the helper models so first use works offline
    helper_files: tuple = ()

    # workbench tools that support this family ("repair", ...); the generic WorkbenchEngine drives them all
    workbench: tuple = ()

    # things a user or a later session must know, with sources
    notes: tuple = ()

    # ---- derived ------------------------------------------------------------------------------
    # generic entry points shared by every described family (the standard layer)
    train_script = "src/fizgig/families/train.py"
    cache_script = "src/fizgig/families/cache.py"

    @property
    def training_ready(self) -> bool:
        return bool(self.driver)

    def load_driver(self):
        """Instantiate the family's FamilyDriver (imported lazily: the description stays importable without torch)."""
        import importlib
        mod, _, cls = self.driver.partition(":")
        drv = getattr(importlib.import_module(mod), cls)()
        drv.description = self
        return drv

    def lora_prefix(self, block: int, module: str) -> str:
        """Key stem of one wrapped module, e.g. 'transformer.transformer_blocks.3.attn.to_q'."""
        return self.lora.key_template.split(".{ab}")[0].format(block=block, module=module)

    @property
    def pref_keys(self) -> tuple:
        return tuple(f.pref_key for f in self.model_files)

    @property
    def required_pref_keys(self) -> tuple:
        return tuple(f.pref_key for f in self.model_files if f.required)

    def pref_for(self, role: str) -> str:
        """Pref key of the model file with this role ("" if the family has none)."""
        return next((f.pref_key for f in self.model_files if f.role == role), "")

    def block_ids(self) -> list:
        return [f"{self.block_prefix}_{i}" for i in range(self.n_blocks)]

    def preview_speed(self):
        """The SpeedLoRA used for in-training previews, or None."""
        return next((sl for sl in self.speed_loras if sl.name == self.preview_speed_lora), None)

    def preview_speed_defaults(self):
        """(steps, strength) previews use with the preview SpeedLoRA, or None without one."""
        sp = self.preview_speed()
        if sp is None:
            return None
        return (self.preview_speed_steps or sp.settings.steps,
                sp.strength if self.preview_speed_strength is None else self.preview_speed_strength)

    def default_sampling(self) -> Optional[SamplingSettings]:
        return self.sampling[0] if self.sampling else None

    def architecture_entry(self) -> dict:
        """The ARCHITECTURES-shaped dict the GUI's existing config readers expect, so start_training,
        the Samples tab and validation can read a description family without KeyErrors. Klein-shaped
        keys that do not apply are filled with the neutral values the Krea 2 entry uses."""
        s = self.default_sampling()
        return {
            "family_key": self.key,
            "is_description_family": True,
            "train_script": self.train_script,
            "cache_latents_script": self.cache_script,
            "cache_text_script": self.cache_script,
            "network_module": "fizgig.networks.lora_klein",   # unused: the family trainer builds its own net
            "use_fizgig_venv": True,
            "timestep_sampling": "shift",
            "discrete_flow_shift": None,
            "weighting_scheme": "none",
            "blocks_swap_max": max(0, self.n_blocks - 2),
            "fp8_text_encoder_flag": None,
            "uses_clip": False,
            "uses_t5": False,
            "uses_text_encoder": True,
            "uses_model_type": False,
            "uses_model_version": False,
            "model_version": self.arch_id,
            "vae_label": self.vae_label,
            "text_encoder_label": self.text_encoder_label,
            "is_distilled": False,
            "supports_weighting_scheme": False,
            "supports_discrete_flow_shift": False,
            "supports_samples": True,
            "sample_cfg_default": self.preview_cfg,
            "sample_flow_shift_default": None,
            "sample_steps_default": self.preview_steps if self.preview_steps else (s.steps if s else 20),
            "sample_width_default": self.preview_width,
            "sample_height_default": self.preview_height,
            "lora_name_suffix": self.lora_name_suffix,
        }

    def validate(self) -> list:
        """Internal consistency problems (empty list = fine). Cheap; run by the registry and tests."""
        problems = []
        if not (self.key and self.arch_id and self.display_name and self.gui_label):
            problems.append("identity fields must all be set")
        keys = [f.pref_key for f in self.model_files]
        if len(keys) != len(set(keys)):
            problems.append("duplicate pref keys")
        if self.n_blocks <= 0 or not self.block_prefix:
            problems.append("block layout missing")
        if self.lora is None:
            problems.append("LoRA format missing")
        if not self.sampling:
            problems.append("no sampling settings")
        if self.bucket_step % self.spatial_factor:
            problems.append("bucket_step must be a multiple of spatial_factor")
        for sl in self.speed_loras:
            if not (sl.repo and sl.file and sl.source):
                problems.append(f"speed LoRA {sl.name!r} is missing repo/file/source")
        if self.driver and ":" not in self.driver:
            problems.append("driver must be 'module.path:ClassName'")
        if self.driver and not all(self.pref_for(r) for r in ("dit", "vae", "text_encoder")):
            problems.append("a trainable family needs model files with roles dit, vae and text_encoder")
        if self.training_adapter and self.pref_for("training_adapter") != self.training_adapter:
            problems.append("training_adapter must name the model file whose role is training_adapter")
        if self.driver and not (self.modelspec_arch and self.implementation):
            problems.append("a trainable family needs modelspec_arch and implementation for LoRA metadata")
        return problems
