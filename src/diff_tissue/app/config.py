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


class OutputManager:
    def __init__(self, output_type_dir: str | None, base_dir: str):
        self._output_type_dir = output_type_dir
        self._base_dir = base_dir

    @property
    def _root(self):
        if self._output_type_dir is None:
            return Path(self._base_dir)
        else:
            return Path(self._base_dir) / self._output_type_dir

    def _prepare(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def file_path(self, *parts: str) -> Path:
        return self._prepare(self._root / Path(*parts))

    def cache_path(self, *parts: str) -> Path:
        return self._prepare(self._root / "cache" / Path(*parts))


class ProjectPaths:
    def __init__(self, data_base_dir, outputs_base_dir):
        self._data_base_dir = Path(data_base_dir)
        self._outputs_base_dir = Path(outputs_base_dir)

    def _prepare(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_base_dir(self):
        return self._prepare(self._data_base_dir)

    @property
    def raw_data_dir(self):
        return self._prepare(self._data_base_dir / "raw")

    @property
    def interim_data_dir(self):
        return self._prepare(self._data_base_dir / "interim")

    @property
    def processed_data_dir(self):
        return self._prepare(self._data_base_dir / "processed")

    @property
    def outputs_base_dir(self):
        return self._prepare(self._outputs_base_dir)

    def grid_search_data_dir(self, study_name):
        return self._prepare(
            self.processed_data_dir / "grid_search" / study_name
        )

    def grid_search_figs_dir(self, study_name):
        return self._prepare(
            self.outputs_base_dir / "grid_search" / study_name
        )
