import json
import pickle

import numpy as np
import yaml

from ..core.jax_bootstrap import jax


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
