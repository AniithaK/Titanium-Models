from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "code" / "train_titanium_model_a_improved_v2.py"

original_module_from_spec = importlib.util.module_from_spec


def registered_module_from_spec(spec):
    module = original_module_from_spec(spec)
    sys.modules[spec.name] = module
    return module


# Dataclasses inspect sys.modules while the dynamically loaded base module executes.
importlib.util.module_from_spec = registered_module_from_spec
spec = importlib.util.spec_from_file_location("improved_training", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.FEATURES = [
    "world_weight_estimated_by_un_flag" if name == "weight_estimated_by_un_flag" else name
    for name in module.FEATURES
]
module.main()
