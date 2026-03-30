from collections import defaultdict
from dataclasses import fields
from itertools import product
import json
import multiprocessing as mp
import re

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from . import config, parameters
from ..core import shape_opt


def _simulate(vars):
    (
        shape,
        trapezoid_angle,
        areas_pot_w,
        anisotropies_pot_w,
        angles_pot_w,
        seed,
    ) = vars

    params = parameters.Params(
        shape=shape,
        trapezoid_angle=trapezoid_angle,
        areas_pot_weight=areas_pot_w,
        anisotropies_pot_weight=anisotropies_pot_w,
        angles_pot_weight=angles_pot_w,
        quiet=True,
        seed=seed,
    )
    sim_states = shape_opt.run(params, short=True)

    return sim_states


def _format_float_to_str(float_):
    rounded_float = round(float_, 8)
    float_str = str(rounded_float)
    if float_str[0] == "-":
        float_str = f"m{float_str[1:]}"
    return float_str.replace(".", "p")


def _worker(trial_vars, output_manager):
    """Run a single trial and save results to a JSON file."""
    shape, tran, arpw, aspw, anpw, seed = trial_vars
    print(
        f"Running with shape={shape}, "
        f"tran={tran}, "
        f"arpw={arpw}, "
        f"aspw={aspw}, "
        f"anpw={anpw}, "
        f"seed={seed}",
    )

    file_path = output_manager.file_path(
        f"shape={shape}__"
        f"tran={_format_float_to_str(tran)}__"
        f"arpw={_format_float_to_str(arpw)}__"
        f"aspw={_format_float_to_str(aspw)}__"
        f"anpw={_format_float_to_str(anpw)}__"
        f"seed={seed}.json"
    )
    if file_path.exists():
        return None

    sim_states = _simulate(trial_vars)
    best_state = shape_opt.get_best_state(sim_states)
    loss = best_state.loss
    valid = all(sim_states.valid)

    result = {
        "shape": shape,
        "trapezoid_angle": float(tran),
        "areas_pot_weight": float(arpw),
        "anisotropies_pot_weight": float(aspw),
        "angles_pot_weight": float(anpw),
        "seed": int(seed),
        "loss": loss,
        "valid": valid,
    }

    with open(file_path, "w") as f:
        json.dump(result, f)

    return result


def run(grid_variables, study_name, n_workers, output_dir):
    grid_values = [
        getattr(grid_variables, f.name) for f in fields(grid_variables)
    ]
    all_trials = list(product(*grid_values))

    output_manager = config.OutputManager(
        f"grid_search/{study_name}/data", output_dir
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
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {json_path}")
                continue

            shape = data["shape"]

            if shape == "trapezoid":
                if data["trapezoid_angle"] > 90:
                    plotting_shape = "trapezoid_obtuse"
                else:
                    plotting_shape = "trapezoid_acute"
            elif shape == "petal":
                plotting_shape = "petal"
            plotting_data[plotting_shape].append(
                (
                    data["trapezoid_angle"],
                    data["areas_pot_weight"],
                    data["anisotropies_pot_weight"],
                    data["loss"],
                    data["valid"],
                )
            )

        data_by_anpw[anpw_str] = plotting_data

    return data_by_anpw


def _add_colorbar(ax, cmap_vals, cmap_name):
    normalize = colors.Normalize(vmin=0.0, vmax=cmap_vals.max())
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array(cmap_vals)
    ax.figure.colorbar(sm, ax=ax, shrink=1.0)


def plot(study_name, outputs_base_dir):
    output_manager = config.OutputManager(
        f"grid_search/{study_name}", outputs_base_dir
    )
    input_dir = output_manager.file_path("data")

    all_files = input_dir.glob("*")
    unique_anpw_val_strs = _find_unique_anpw_val_strs(all_files)

    data_by_anpw = _get_plotting_data(unique_anpw_val_strs, input_dir)

    cmap_name = "RdYlGn_r"

    ordered_shapes = ["trapezoid_acute", "trapezoid_obtuse", "petal"]
    n_plots = len(ordered_shapes)

    for anpw_str, plotting_data in data_by_anpw.items():
        fig, axs = plt.subplots(n_plots, constrained_layout=True)

        for i, shape in enumerate(ordered_shapes):
            data_list_of_tuples = plotting_data.get(shape)
            if data_list_of_tuples is None:
                continue
            ax = axs[i]
            arpw_vals = np.array([tup[1] for tup in data_list_of_tuples])
            aspw_vals = np.array([tup[2] for tup in data_list_of_tuples])
            losses = np.array([tup[3] for tup in data_list_of_tuples])

            valid = np.array([tup[4] for tup in data_list_of_tuples])
            valid_losses = losses[valid]

            if len(valid_losses) > 0:
                ax.scatter(
                    arpw_vals[valid],
                    aspw_vals[valid],
                    c=valid_losses,
                    cmap=cmap_name,
                )
                _add_colorbar(ax, valid_losses, cmap_name)

            ax.scatter(arpw_vals[~valid], aspw_vals[~valid], marker="x", c="k")
            ax.set_title(f"{shape}")
            if i != n_plots - 1:
                ax.set_xticklabels([])
            if i == n_plots - 1:
                ax.set_xlabel("Area pot. weights")
            ax.set_ylabel("Anisotropy\n pot. weights")

        fig_path = output_manager.file_path("figures", f"{anpw_str}.pdf")
        fig.savefig(fig_path)
        plt.close(fig)
