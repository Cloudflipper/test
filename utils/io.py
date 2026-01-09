"""I/O utilities for loading and saving configurations."""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge model config into base config."""
    merged = base_config.copy()
    
    # Deep merge for nested dictionaries
    for key, value in model_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

