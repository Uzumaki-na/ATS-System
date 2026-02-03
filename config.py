import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.environ.get("TRIADRANK_CONFIG_PATH")

    if config_path is None:
        config_path = Path(__file__).with_name("config.yaml")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid config format in {config_path}. Expected a YAML mapping.")

    return loaded


config: Dict[str, Any] = load_config()
