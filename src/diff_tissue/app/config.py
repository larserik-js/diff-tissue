from dataclasses import dataclass
from pathlib import Path

from . import io_utils


@dataclass
class _Config:
    data_base_dir: str
    outputs_base_dir: str


def load_cfg(path):
    data = io_utils.load_yaml(path)
    config = _Config(**data)
    return config


class ProjectPaths:
    def __init__(self, data_base_dir, outputs_base_dir):
        self.data_base_dir = Path(data_base_dir)
        self.outputs_base_dir = Path(outputs_base_dir)

    @property
    def raw_data_dir(self):
        return self.data_base_dir / "raw"

    @property
    def interim_data_dir(self):
        return self.data_base_dir / "interim"

    @property
    def processed_data_dir(self):
        return self.data_base_dir / "processed"
