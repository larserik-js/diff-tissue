from dataclasses import fields
from itertools import product
import json
import multiprocessing as mp

from . import io_utils, parameters
from ..core import init_systems, metrics, shape_opt


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

    return best_state.loss, n_edge_crossings


def _worker(trial_vars, output_manager):
    """Run a single trial and save results to a JSON file."""
    shape, arpw, aspw, anpw = trial_vars
    loss, n_edge_crossings = _simulate(trial_vars)

    result = {
        "shape": shape,
        "areas_pot_weight": float(arpw),
        "anisotropies_pot_weight": float(aspw),
        "angles_pot_weight": float(anpw),
        "loss": loss,
        "n_edge_crossings": n_edge_crossings,
    }

    file_path = output_manager.file_path(
        f"shape={shape}__arpw={arpw}__aspw={aspw}__anpw={anpw}.json"
    )
    with open(file_path, "w") as f:
        json.dump(result, f)

    return result


def run(grid_variables, study_name, n_workers):
    grid_values = [
        getattr(grid_variables, f.name) for f in fields(grid_variables)
    ]
    all_trials = list(product(*grid_values))

    output_manager = io_utils.OutputManager(
        f"grid_search/{study_name}", "outputs"
    )

    inputs = [(trial, output_manager) for trial in all_trials]

    results = []
    with mp.Pool(processes=n_workers) as pool:
        for result in pool.starmap(_worker, inputs):
            results.append(result)
            completed = len(results)
            print(
                f"Completed {completed}/{len(all_trials)} trials\n",
                flush=True,
            )

    print("\nAll trials completed.")
