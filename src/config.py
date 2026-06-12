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

# API keys for FRED); not committed to version control
API_KEYS = cfg.get("api_keys", {})

# Shared gross-return grid for Phases 2, 3, and 5
RETURN_GRID = cfg.get('return_grid', {'min': 0.40, 'max': 2.00, 'points': 1000})

def get_return_grid():
    import numpy as np
    return np.linspace(float(RETURN_GRID['min']), float(RETURN_GRID['max']),
                       int(RETURN_GRID['points']))