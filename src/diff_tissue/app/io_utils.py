from dataclasses import asdict
import json
from pathlib import Path
import pickle

import numpy as np
import yaml


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_dict_of_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return data


def save_arrays(path: Path, **arrays_by_name) -> None:
    np.savez(path, **arrays_by_name)


def save_arrays_from_dataclass(path, dataclass):
    save_arrays(path, **asdict(dataclass))


def save_pdf(path, fig, dpi=None):
    if dpi is not None:
        fig.savefig(path, dpi=dpi)
    else:
        fig.savefig(path)


def load_yaml(path):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
