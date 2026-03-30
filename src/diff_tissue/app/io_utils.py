from pathlib import Path
import pickle

import numpy as np
import yaml

from ..core.jax_bootstrap import jax


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


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def save_arrays(path, arrays):
    arrays_np = jax.device_get(arrays)
    np.savez(path, arrays=arrays_np)


def load_arrays(path):
    data = np.load(path)
    return data["arrays"]


def load_yaml(path):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg
