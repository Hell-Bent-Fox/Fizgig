"""Plug-in model families: one FamilyDescription per family, read by the GUI's generic paths.

See description.py. Existing families (Klein, Krea 2, MiniMax H3) are not described here by design.
"""
from fizgig.families.description import (  # noqa: F401
    FamilyDescription, LoRAFormat, ModelFile, SamplingSettings, SpeedLoRA,
)
from fizgig.families.registry import FAMILIES, by_gui_label, get, training_families  # noqa: F401
