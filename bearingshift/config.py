"""Load YAML configs. A config may name a `base` config and override only some keys."""
from pathlib import Path

import yaml


def deep_update(base, overrides):
    out = dict(base)
    for key, value in overrides.items():
        out[key] = deep_update(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("base", None)
    return deep_update(load_config(path.parent / base), cfg) if base else cfg
