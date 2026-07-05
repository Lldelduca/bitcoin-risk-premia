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
    key = f"filters_{dataset}"
    
    if key not in cfg:
        raise KeyError(f"Filter key '{key}' not found in config.yaml")
        
    return cfg[key]

def get_sample_window() -> tuple[str, str]:
    cfg = load_config()
    sample = cfg.get("sample", {})
    return sample.get("start_date"), sample.get("end_date")