from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "code" / "train_titanium_model_a_improved_v2.py"
spec = importlib.util.spec_from_file_location("improved_training", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# The source data uses this exact UN Comtrade audit-column name.
module.FEATURES = [
    "world_weight_estimated_by_un_flag" if name == "weight_estimated_by_un_flag" else name
    for name in module.FEATURES
]
module.main()
