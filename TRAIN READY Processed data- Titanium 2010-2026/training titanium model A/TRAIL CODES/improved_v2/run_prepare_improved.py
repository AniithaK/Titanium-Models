from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "prepare_improved_training_data.py"
spec = importlib.util.spec_from_file_location("improved_preparation", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

original_read_csv = module.pd.read_csv


def read_csv_with_source_alias(*args, **kwargs):
    frame = original_read_csv(*args, **kwargs)
    if "world_weight_estimated_by_un_flag" in frame and "weight_estimated_by_un_flag" not in frame:
        frame["weight_estimated_by_un_flag"] = frame["world_weight_estimated_by_un_flag"]
    return frame


module.pd.read_csv = read_csv_with_source_alias
module.main()
