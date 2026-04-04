from collections import defaultdict
from dataclasses import fields
from itertools import product
import json
import multiprocessing as mp
import re

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from . import io_utils, parameters
from ..core import shape_opt


def _simulate(vars):
    (
        shape,
        knots,
        trapezoid_angle,
        areas_pot_w,
        anisotropies_pot_w,
        angles_pot_w,
        seed,
    ) = vars

    params = parameters.Params(
        shape=shape,
        knots=knots,
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


def _worker(trial_vars, output_dir):
    """Run a single trial and save results to a JSON file."""
    shape, knots, tran, arpw, aspw, anpw, seed = trial_vars
    print(
        f"Running with shape={shape}, "
        f"knots={knots}, "
        f"tran={tran}, "
        f"arpw={arpw}, "
        f"aspw={aspw}, "
        f"anpw={anpw}, "
        f"seed={seed}",
    )

    file_path = output_dir / (
        f"shape={shape}__"
        f"knots={knots}__"
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
        "knots": knots,
        "trapezoid_angle": float(tran),
        "areas_pot_weight": float(arpw),
        "anisotropies_pot_weight": float(aspw),
        "angles_pot_weight": float(anpw),
        "seed": int(seed),
        "loss": loss,
        "valid": valid,
    }

    io_utils.save_json(file_path, result)

    return result


def run(grid_variables, study_name, n_workers, paths):
    grid_values = [
        getattr(grid_variables, f.name) for f in fields(grid_variables)
    ]
    all_trials = list(product(*grid_values))

    output_dir = paths.grid_search_data_dir(study_name)

    inputs = [(trial, output_dir) for trial in all_trials]

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
    pattern = re.compile(r"anpw=(.*?)__")
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
                data = io_utils.load_json(json_path)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {json_path}")
                continue

            shape = data["shape"]

            if shape == "trapezoid":
                if np.isclose(data["trapezoid_angle"], 100.0):
                    continue
                elif np.isclose(data["trapezoid_angle"], 80.0):
                    plotting_shape = "wide_trapezoid"
                elif np.isclose(data["trapezoid_angle"], 90.0):
                    plotting_shape = "square"
                else:
                    raise ValueError(
                        f"Unexpected trapezoid angle: {data['trapezoid_angle']}"
                    )
            elif shape == "petal":
                plotting_shape = "petal"
            elif shape == "nconv":
                plotting_shape = "nconv"
            else:
                raise ValueError(f"Unexpected shape: {shape}")

            strict_valid = data["valid"]

            plotting_data[plotting_shape].append(
                (
                    data["trapezoid_angle"],
                    data["areas_pot_weight"],
                    data["anisotropies_pot_weight"],
                    data["loss"],
                    strict_valid,
                )
            )

        data_by_anpw[anpw_str] = plotting_data

    return data_by_anpw


def _calc_global_loss_bounds(data_by_anpw):
    all_losses = []
    for plotting_data in data_by_anpw.values():
        for shape_data in plotting_data.values():
            for tup in shape_data:
                if tup[4]:  # Only consider valid runs for loss bounds
                    all_losses.append(tup[3])
    return (0.0, max(all_losses))


def _add_colorbar(ax, normalize_fn, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(norm=normalize_fn, cmap=cmap)
    ax.figure.colorbar(sm, ax=ax, shrink=1.0)


def _find_ax_limits(data_by_anpw):
    all_arpw_vals = []
    all_aspw_vals = []

    for plotting_data in data_by_anpw.values():
        for shape_data in plotting_data.values():
            for tup in shape_data:
                all_arpw_vals.append(tup[1])
                all_aspw_vals.append(tup[2])

    arpw_min, arpw_max = min(all_arpw_vals), max(all_arpw_vals)
    aspw_min, aspw_max = min(all_aspw_vals), max(all_aspw_vals)
    offset = 2.0

    return (arpw_min - offset, arpw_max + offset), (
        aspw_min - offset,
        aspw_max + offset,
    )


def plot(study_name, paths):
    input_dir = paths.grid_search_data_dir(study_name)

    all_files = input_dir.glob("*")
    unique_anpw_val_strs = _find_unique_anpw_val_strs(all_files)

    data_by_anpw = _get_plotting_data(unique_anpw_val_strs, input_dir)

    cmap_name = "RdYlGn_r"

    ordered_shapes = [
        "wide_trapezoid",
        "square",
        "petal",
        "nconv",
    ]
    n_plots = len(ordered_shapes)

    output_dir = paths.grid_search_figs_dir(study_name)

    global_loss_bounds = _calc_global_loss_bounds(data_by_anpw)
    normalize_loss = colors.Normalize(
        vmin=global_loss_bounds[0], vmax=global_loss_bounds[1]
    )
    ax_lims = _find_ax_limits(data_by_anpw)

    n_rows = int(np.ceil(n_plots / 2))
    n_cols = 2 if n_plots > 1 else 1

    for anpw_str, plotting_data in data_by_anpw.items():
        fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True)

        for k, shape in enumerate(ordered_shapes):
            data_list_of_tuples = plotting_data.get(shape)
            if data_list_of_tuples is None:
                continue
            i, j = divmod(k, n_cols)
            ax = axs[i, j]
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
                    s=10.0,
                    marker="s",
                )
                _add_colorbar(ax, normalize_loss, cmap_name)

            ax.scatter(
                arpw_vals[~valid], aspw_vals[~valid], s=1.0, marker="x", c="k"
            )
            ax.set_xlim(ax_lims[0])
            ax.set_ylim(ax_lims[1])

            ax.set_title(f"{shape}")
            if i != n_rows - 1:
                ax.set_xticklabels([])
            if i == n_rows - 1:
                ax.set_xlabel("Area pot. weights")
            if j == 0:
                ax.set_ylabel("Anisotropy\n pot. weights")
            else:
                ax.set_yticklabels([])

        fig_path = output_dir / f"anpw={anpw_str}.pdf"
        io_utils.save_pdf(fig_path, fig)
        plt.close(fig)
