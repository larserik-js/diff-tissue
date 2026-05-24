from dataclasses import asdict

import numpy as np

from . import parameters
from ..core import shape_opt


def _sim_states_to_dict(sim_states: shape_opt.SimStates) -> dict:
    """
    Convert SimStates dataclass to a JSON-serializable dictionary.
    Converts all NumPy arrays to Python lists.
    """

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    return {k: convert(v) for k, v in asdict(sim_states).items()}


def run_and_serialize() -> dict:
    params = parameters.Params()
    sim_states = shape_opt.run(params)
    serialized_sim_states = _sim_states_to_dict(sim_states)
    return serialized_sim_states
