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

    def make_subdir(self, *parts):
        subdir = Path(*parts)
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    @property
    def sim_states_dir(self):
        sim_states_dir = self.make_subdir(
            self.processed_data_dir / "sim_states"
        )
        return sim_states_dir

    @property
    def param_search_db(self):
        data_dir = self.make_subdir(self.interim_data_dir)
        db_path = data_dir / "optuna.db"
        return db_path
