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


def run_and_serialize(request_data) -> dict:
    params = parameters.Params(
        system=request_data.system,
        shape=request_data.shape,
        knots=request_data.knots,
        quiet=request_data.quiet,
        trapezoid_angle=request_data.trapezoid_angle,
        n_morph_steps=request_data.n_morph_steps,
        areas_pot_weight=request_data.areas_pot_weight,
        anisotropies_pot_weight=request_data.anisotropies_pot_weight,
        angles_pot_weight=request_data.angles_pot_weight,
        init_lr=request_data.init_lr,
        n_shape_steps=request_data.n_shape_steps,
        shape_loss_weight=request_data.shape_loss_weight,
        var_loss_weight=request_data.var_loss_weight,
        poly_id_cfg=request_data.poly_id_cfg,
        seed=request_data.seed,
    )
    sim_states = shape_opt.run(params)
    serialized_sim_states = _sim_states_to_dict(sim_states)
    return serialized_sim_states
