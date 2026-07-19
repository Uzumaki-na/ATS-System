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

    _normalize_category_keys(loaded)
    return loaded


def _normalize_category_keys(cfg: Dict[str, Any]) -> None:
    """Ensure categories.id_to_name keys and name_to_id values are int (B8).

    PyYAML parses unquoted integer keys as int but quoted keys as str, so the two
    consumers that index by int (api/main.py) and the one that does int(k)
    defensively (run.py) disagreed on the assumed key type. Normalize once here so
    every consumer can index id_to_name by int safely.
    """
    cats = cfg.get("categories")
    if not isinstance(cats, dict):
        return
    id_to_name = cats.get("id_to_name")
    if isinstance(id_to_name, dict):
        cats["id_to_name"] = {int(k): v for k, v in id_to_name.items()}
    name_to_id = cats.get("name_to_id")
    if isinstance(name_to_id, dict):
        cats["name_to_id"] = {k: int(v) for k, v in name_to_id.items()}


config: Dict[str, Any] = load_config()

