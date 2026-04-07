from dataclasses import fields
from itertools import product
import json
import multiprocessing as mp

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import config, io_utils, parameters
from ..core import shape_opt


class GridSearchPaths(config.ProjectPaths):
    def __init__(self, base_paths, study_name):
        super().__init__(
            data_base_dir=base_paths.data_base_dir,
            outputs_base_dir=base_paths.outputs_base_dir,
        )
        self.study_name = study_name

    @property
    def individual_results_dir(self):
        return self.make_subdir(
            self.interim_data_dir / "grid_search" / self.study_name
        )

    @property
    def tabular_results_dir(self):
        return self.make_subdir(
            self.processed_data_dir / "grid_search" / self.study_name
        )

    @property
    def tabular_results_path(self):
        return self.tabular_results_dir / "results.parquet"

    @property
    def figures_dir(self):
        return self.make_subdir(
            self.outputs_base_dir / "grid_search" / self.study_name
        )


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
        f"tran={parameters.format_float_to_str(tran)}__"
        f"arpw={parameters.format_float_to_str(arpw)}__"
        f"aspw={parameters.format_float_to_str(aspw)}__"
        f"anpw={parameters.format_float_to_str(anpw)}__"
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


def _individual_results_to_df(input_dir, output_path):
    rows = []

    for file in input_dir.glob("*.json"):
        with open(file) as f:
            rows.append(json.load(f))

    df = pl.DataFrame(rows)
    df.write_parquet(output_path)


def run(grid_variables, study_name, n_workers, paths):
    grid_values = [
        getattr(grid_variables, f.name) for f in fields(grid_variables)
    ]
    all_trials = list(product(*grid_values))

    grid_search_paths = GridSearchPaths(paths, study_name)

    inputs = [
        (trial, grid_search_paths.individual_results_dir)
        for trial in all_trials
    ]

    results = []
    with mp.Pool(processes=n_workers) as pool:
        print(f"Running {len(all_trials)} trials with {n_workers} workers...")
        for result in pool.starmap(_worker, inputs):
            results.append(result)

    print("All trials completed.")
    print("")


def _shape_to_plotting_shape(trapezoid_angle, shape):
    if shape == "trapezoid":
        if np.isclose(trapezoid_angle, 80.0):
            plotting_shape = "wide_trapezoid"
        elif np.isclose(trapezoid_angle, 90.0):
            plotting_shape = "square"
        elif np.isclose(trapezoid_angle, 100.0):
            plotting_shape = "narrow_trapezoid"
        else:
            raise ValueError(f"Unexpected trapezoid angle: {trapezoid_angle}")
    else:
        plotting_shape = shape
    return plotting_shape


def _transform_df(df):
    df = df.with_columns(
        pl.struct(["trapezoid_angle", "shape"])
        .map_elements(
            lambda x: _shape_to_plotting_shape(
                x["trapezoid_angle"], x["shape"]
            ),
            return_dtype=pl.Utf8,
        )
        .alias("plotting_shape")
    )
    df = df.drop("shape")
    df = df.with_columns(
        pl.col("angles_pot_weight")
        .map_elements(parameters.format_float_to_str)
        .alias("angles_pot_weight")
    )
    return df


def _get_plotting_data(df):
    group_cols = ["angles_pot_weight", "plotting_shape"]
    value_cols = value_cols = [c for c in df.columns if c not in group_cols]
    result = df.group_by(group_cols).agg(
        pl.struct(value_cols).alias("row_dicts")
    )

    data_by_anpw: dict[str, dict] = dict()

    for row in result.iter_rows(named=True):
        angle = row["angles_pot_weight"]
        shape = row["plotting_shape"]
        dict_list = [dict(d) for d in row["row_dicts"]]

        data_by_anpw.setdefault(angle, {})[shape] = dict_list

    return data_by_anpw


def _calc_global_loss_bounds(data_by_anpw):
    all_losses = []
    for plotting_data in data_by_anpw.values():
        for shape_data in plotting_data.values():
            for dict_ in shape_data:
                if dict_["valid"]:  # Only consider valid runs for loss bounds
                    all_losses.append(dict_["loss"])
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
            for dict_ in shape_data:
                all_arpw_vals.append(dict_["areas_pot_weight"])
                all_aspw_vals.append(dict_["anisotropies_pot_weight"])

    arpw_min, arpw_max = min(all_arpw_vals), max(all_arpw_vals)
    aspw_min, aspw_max = min(all_aspw_vals), max(all_aspw_vals)
    offset = 2.0

    return (arpw_min - offset, arpw_max + offset), (
        aspw_min - offset,
        aspw_max + offset,
    )


def _get_df(grid_search_paths):
    df_file = grid_search_paths.tabular_results_path
    if not df_file.exists():
        print("Converting individual JSON results to Parquet...")
        _individual_results_to_df(
            grid_search_paths.individual_results_dir,
            grid_search_paths.tabular_results_path,
        )
        print("Conversion completed.")
    df = pl.read_parquet(df_file)
    return df


def plot(study_name, paths):
    grid_search_paths = GridSearchPaths(paths, study_name)

    df = _get_df(grid_search_paths)

    df = _transform_df(df)

    data_by_anpw = _get_plotting_data(df)

    cmap_name = "RdYlGn_r"

    ordered_shapes = [
        "wide_trapezoid",
        "square",
        "petal",
        "nconv",
    ]
    n_plots = len(ordered_shapes)

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
            data_list_of_dicts = plotting_data.get(shape)
            if data_list_of_dicts is None:
                continue
            i, j = divmod(k, n_cols)
            ax = axs[i, j]
            arpw_vals = np.array(
                [dict_["areas_pot_weight"] for dict_ in data_list_of_dicts]
            )
            aspw_vals = np.array(
                [
                    dict_["anisotropies_pot_weight"]
                    for dict_ in data_list_of_dicts
                ]
            )
            losses = np.array([dict_["loss"] for dict_ in data_list_of_dicts])

            valid = np.array([dict_["valid"] for dict_ in data_list_of_dicts])
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

        fig_path = grid_search_paths.figures_dir / f"anpw={anpw_str}.pdf"
        io_utils.save_pdf(fig_path, fig)
        plt.close(fig)
