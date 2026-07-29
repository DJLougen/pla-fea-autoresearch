from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import yaml


REQUIRED_SECTIONS = {"experiment", "solver", "data", "optimization", "output"}
ALLOWED_MODEL_CLASSES = {"isotropic", "transverse_isotropic", "orthotropic"}
ALLOWED_SOLVER_MODES = {"calculix", "mock"}


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    config = deepcopy(config)
    if config["experiment"].get("model_class") not in ALLOWED_MODEL_CLASSES:
        raise ValueError("unsupported model_class")
    if config["solver"].get("mode") not in ALLOWED_SOLVER_MODES:
        raise ValueError("solver.mode must be calculix or mock")
    timeout = float(config["solver"].get("timeout_seconds", 0))
    if not 0 < timeout <= 3600:
        raise ValueError("timeout_seconds must be in (0, 3600]")
    if int(config["optimization"].get("budget", 0)) <= 0:
        raise ValueError("optimization budget must be positive")
    config["_config_path"] = str(path.resolve())
    return config


def dump_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False)
