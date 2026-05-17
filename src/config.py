import yaml
from pathlib import Path

# Identify the root directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / 'config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

PATHS = cfg['paths']

# Sample window
SAMPLE = cfg['sample']

# Shared filters (apply to both venues)
FILTERS_SHARED = cfg['filters_shared']

# Venue-specific filters
FILTERS_CME = {**FILTERS_SHARED, **cfg['filters_cme']}
FILTERS_DERIBIT = {**FILTERS_SHARED, **cfg['filters_deribit']}

# SSVI fitting parameters
SSVI_CONFIG = cfg['ssvi']

# Tensor PCA grid (consumed by Phase 1b, not Phase 1a)
TENSOR_GRID = cfg.get('tensor_grid', {})

def get_path(path_key: str) -> Path:
    """Returns absolute path from config key."""
    return BASE_DIR / PATHS[path_key]