from itertools import product

import numpy as np
import optuna

from . import io_utils, parameters
from ..core import init_systems, metrics, shape_opt


def _calc_n_total_trials(
    area_pot_loss_vals, anisotropy_pot_loss_vals, angle_pot_loss_vals
):
    n_total_runs = (
        len(area_pot_loss_vals)
        * len(anisotropy_pot_loss_vals)
        * len(angle_pot_loss_vals)
    )
    return n_total_runs


def _round(var, digits=3):
    return round(var, digits)


def _simulate(vars):
    shape, areas_pot_w, anisotropies_pot_w, angles_pot_w = vars

    params = parameters.Params(
        shape=shape,
        areas_pot_weight=areas_pot_w,
        anisotropies_pot_weight=anisotropies_pot_w,
        angles_pot_weight=angles_pot_w,
        quiet=True,
    )
    sim_states = shape_opt.run(params)
    best_state = shape_opt.get_best_state(sim_states)

    polygon_inds = init_systems.get_system(params).indices
    n_edge_crossings = metrics.count_edge_crossings(
        best_state.final_vertices, polygon_inds
    )

    if n_edge_crossings > 0:
        loss = np.inf
    else:
        loss = best_state.loss
    return loss


def _enqueue_trials(vars, study):
    for shape, arpw, aspw, anpw in product(*vars):
        arpw, aspw, anpw = map(_round, (arpw, aspw, anpw))
        trial_params = {
            "shape": shape,
            "areas_pot_weight": arpw,
            "anosotropies_pot_weight": aspw,
            "angles_pot_weight": anpw,
        }
        study.enqueue_trial(trial_params)


def _optimize(study, objective_f, n_total_trials):
    def _progress_callback(study_, trial):
        completed_trials = len(
            [
                t
                for t in study_.get_trials(deepcopy=False)
                if t.state.is_finished()
            ]
        )
        print(
            f"Progress: {completed_trials}/{n_total_trials} trials completed"
        )

    study.optimize(
        objective_f,
        n_trials=n_total_trials,
        n_jobs=20,
        callbacks=[_progress_callback],
    )


def run(shapes, areas_pot_ws, anisotropies_pot_ws, angles_pot_ws):
    def _objective(trial):
        shape = trial.suggest_categorical("shape", shapes)
        arpw = trial.suggest_categorical("areas_pot_weight", areas_pot_ws)
        aspw = trial.suggest_categorical(
            "anisotropies_pot_weight", anisotropies_pot_ws
        )
        anpw = trial.suggest_categorical("angles_pot_weight", angles_pot_ws)

        arpw, aspw, anpw = map(_round, (arpw, aspw, anpw))

        vars = (shape, arpw, aspw, anpw)

        loss = _simulate(vars)
        return loss

    output_manager = io_utils.OutputManager(None, base_dir="outputs")
    db_path = output_manager.file_path("grid_search.db")
    db_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="grid_search",
        storage=db_url,
        load_if_exists=True,
        direction="minimize",
    )

    vars = (shapes, areas_pot_ws, anisotropies_pot_ws, areas_pot_ws)

    _enqueue_trials(vars, study=study)

    n_total_trials = _calc_n_total_trials(
        areas_pot_ws, anisotropies_pot_ws, angles_pot_ws
    )
    _optimize(study, _objective, n_total_trials)
