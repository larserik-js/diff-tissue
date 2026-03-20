from collections import defaultdict
from dataclasses import fields
from itertools import product
import json
import multiprocessing as mp
import re

import matplotlib.pyplot as plt
import numpy as np

from . import io_utils, parameters
from ..core import init_systems, metrics, shape_opt


def _calc_n_edge_crossings(sim_states, valid_inds):
    all_final_vertices = sim_states.final_vertices
    n_edge_crossings = [
        metrics.count_edge_crossings(final_vertices, valid_inds)
        for final_vertices in all_final_vertices
    ]
    return n_edge_crossings


def _calc_edge_crossings_ratio(n_edge_crossings):
    return float((np.array(n_edge_crossings) > 0).mean())


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
    valid_inds = init_systems.make_poly_idx_lists(polygon_inds)

    n_edge_crossings = _calc_n_edge_crossings(sim_states, valid_inds)

    return best_state.loss, n_edge_crossings


def _format_float(float_):
    rounded_float = round(float_, 8)
    float_str = str(rounded_float)
    if float_str[0] == "-":
        float_str = f"m{float_str[1:]}"
    return float_str.replace(".", "p")


def _format_str_to_float(str_):
    float_str = str_.replace("p", ".")
    if float_str[0] == "m":
        float_str = f"-{float_str[1:]}"
    return float(float_str)


def _worker(trial_vars, output_manager):
    """Run a single trial and save results to a JSON file."""
    shape, arpw, aspw, anpw = trial_vars

    file_path = output_manager.file_path(
        f"shape={shape}__arpw={_format_float(arpw)}__"
        f"aspw={_format_float(aspw)}__anpw={_format_float(anpw)}.json"
    )
    if file_path.exists():
        return None

    loss, n_edge_crossings = _simulate(trial_vars)

    result = {
        "shape": shape,
        "areas_pot_weight": float(arpw),
        "anisotropies_pot_weight": float(aspw),
        "angles_pot_weight": float(anpw),
        "loss": loss,
        "n_edge_crossings": n_edge_crossings,
    }

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

    print("All trials completed.")


def _find_unique_anpw_val_strs(all_files):
    pattern = re.compile(r"anpw=(.*?).json")
    all_anpw_val_strs = []

    for file in all_files:
        if file.is_file():
            match = pattern.search(file.name)
            if match:
                anpw_str = match.group(1)
                all_anpw_val_strs.append(anpw_str)
    return list(set(all_anpw_val_strs))


def _get_plotting_data(unique_anpw_val_strs, input_dir):
    data_by_anpw = dict()

    for anpw_str in unique_anpw_val_strs:
        target_str = f"anpw={anpw_str}"
        files = [p for p in input_dir.iterdir() if target_str in p.name]

        plotting_data = defaultdict(list)
        for json_path in files:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            shape = data["shape"]
            plotting_data[shape].append(
                (
                    data["areas_pot_weight"],
                    data["anisotropies_pot_weight"],
                    data["loss"],
                    _calc_edge_crossings_ratio(data["n_edge_crossings"]),
                )
            )

        data_by_anpw[anpw_str] = plotting_data

    return data_by_anpw


def plot(study_name):
    output_manager = io_utils.OutputManager(
        f"grid_search/{study_name}", "outputs"
    )
    input_dir = output_manager.file_path(".")

    all_files = input_dir.glob("*")
    unique_anpw_val_strs = _find_unique_anpw_val_strs(all_files)

    data_by_anpw = _get_plotting_data(unique_anpw_val_strs, input_dir)

    shapes_to_inds = {
        "petal": (0, 0),
        "trapezoid": (0, 1),
        "triangle": (1, 0),
        "nconv": (1, 1),
    }

    for anpw_str, plotting_data in data_by_anpw.items():
        fig, axs = plt.subplots(2, 2)

        for shape, data_list_of_tuples in plotting_data.items():
            ax = axs[shapes_to_inds[shape]]
            data_array = np.vstack(data_list_of_tuples)
            arpw_vals = data_array[:, 0]
            aspw_vals = data_array[:, 1]
            losses = data_array[:, 2]
            markers = np.where(data_array[:, 3] > 0.0, "x", "o")
            for xi, yi, vi, mi in zip(arpw_vals, aspw_vals, losses, markers):
                ax.scatter(
                    xi,
                    yi,
                    c=vi,
                    cmap="viridis",
                    s=100,
                    marker=mi,
                    edgecolor="k",
                )

        fig_path = output_manager.file_path("figures", f"{anpw_str}.pdf")
        fig.savefig(fig_path)
