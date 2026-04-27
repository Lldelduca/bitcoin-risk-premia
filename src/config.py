import yaml
from pathlib import Path

# Identify the root directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / 'config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

PATHS = cfg['paths']
FILTERS = cfg['filters']

def get_path(path_key):
    """Returns absolute path from config key."""
    return BASE_DIR / PATHS[path_key]