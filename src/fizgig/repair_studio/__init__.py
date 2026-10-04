"""LoRA Repair Studio — live per-block LoRA tweaking with side-by-side preview.

Public API:
    SliderState, BlockState — slider configuration data model.
    save_repaired_lora() — bake slider state into a new .safetensors.
The engines: families/workbench.py (WorkbenchEngine, every image family) and h3_engine.py (MiniMax H3).
"""

from fizgig.repair_studio.state import BlockState, SliderState
from fizgig.repair_studio.bake import save_repaired_lora

__all__ = ["BlockState", "SliderState", "save_repaired_lora"]
