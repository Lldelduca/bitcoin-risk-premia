from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
_CONFIG_CACHE = None

def load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE

def get_path(key: str) -> Path:
    cfg = load_config()
    paths = cfg.get("paths", {})
    
    if key not in paths:
        raise KeyError(f"Path key '{key}' not found under 'paths' in config.yaml")
        
    return PROJECT_ROOT / paths[key]

def get_filters(dataset: str = "shared") -> dict:
    cfg = load_config()
    shared = cfg.get("filters_shared", {})
    
    if dataset == "shared":
        return shared
        
    key = f"filters_{dataset}"
    if key not in cfg:
        raise KeyError(f"Filter key '{key}' not found in config.yaml")
        
    return {**shared, **cfg[key]}

def get_sample_window() -> tuple[str, str]:
    cfg = load_config()
    sample = cfg.get("sample", {})
    return sample.get("start_date"), sample.get("end_date")

def get_api_key(service: str) -> str:
    cfg = load_config()
    return cfg.get("api_keys", {}).get(service)

# Phase 1-5 Specific Configurations
def get_ssvi_config() -> dict:
    return load_config().get("ssvi", {})

def get_tensor_grid() -> dict:
    return load_config().get("tensor_grid", {})

def get_return_grid():
    import numpy as np
    cfg = load_config()
    grid_cfg = cfg.get('return_grid', {'min': 0.40, 'max': 2.00, 'points': 1000})
    
    return np.linspace(
        float(grid_cfg['min']), 
        float(grid_cfg['max']),
        int(grid_cfg['points'])
    )