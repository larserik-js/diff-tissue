from dataclasses import dataclass

from . import io_utils


@dataclass
class _Config:
    outputs_base_dir: str


def load_cfg(path):
    data = io_utils.load_yaml(path)
    config = _Config(**data)
    return config
