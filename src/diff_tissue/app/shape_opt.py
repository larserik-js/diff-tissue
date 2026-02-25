import pandas as pd

from ..core import my_utils
from ..core import shape_opt as shape_opt_core
from . import io_utils, morphing, parameters, plotting


OUTPUT_TYPE_DIR = "shape_opt"
FINAL_TISSUES_DIR = "final_tissues"
BEST_GROWTH_DIR = "best_growth"


def _save_output_params(tabular_output, params):
    df = pd.DataFrame(tabular_output)
    output_file = io_utils.get_output_params_file(params)
    df.to_csv(output_file, sep="\t", index=True, header=True)


def load_output_params(params):
    input_file = io_utils.get_output_params_file(params)
    df = pd.read_csv(input_file, sep="\t", index_col=0)

    best_goal_areas = my_utils.to_jax(df["best_goal_area"].values)
    best_goal_anisotropies = my_utils.to_jax(df["best_goal_anisotropy"].values)
    return best_goal_areas, best_goal_anisotropies


def _plot_final_tissues(final_tissues, output, param_string, params):
    figure = plotting.MorphFigure(params)

    for t, vertices in enumerate(final_tissues):
        if t % 10 == 0:
            fig_path = output.file_path(
                f"{FINAL_TISSUES_DIR}", param_string, f"step={t:03d}.png"
            )
            figure.save_plot(vertices, fig_path, enumerate=True)
    fig_path = output.file_path(
        f"{FINAL_TISSUES_DIR}", param_string, f"step={t:03d}.png"
    )
    figure.save_plot(vertices, fig_path, enumerate=True)


def optimize_shape(params, output):
    param_string = parameters.get_param_string(params)
    cache_path = output.cache_path(f"final_tissues__{param_string}.pkl")

    _, final_tissues, _, tabular_output = shape_opt_core.run(params)

    _save_output_params(tabular_output, params)

    io_utils.save_pkl(cache_path, final_tissues)


def plot_final_tissues(params, output):
    param_string = parameters.get_param_string(params)
    cache_path = output.cache_path(f"final_tissues__{param_string}.pkl")
    if cache_path.exists():
        final_tissues = io_utils.load_pkl(cache_path)
    else:
        _, final_tissues, _, tabular_output = shape_opt_core.run(params)

        _save_output_params(tabular_output, params)

        io_utils.save_pkl(cache_path, final_tissues)

    _plot_final_tissues(final_tissues, output, param_string, params)


def get_best_growth_evolution(
    best_goal_areas, best_goal_anisotropies, jax_arrays, params, cache_path
):
    if cache_path.exists():
        best_growth_evolution = io_utils.load_pkl(cache_path)
    else:
        best_growth_evolution = morphing.jiterate(
            best_goal_areas,
            best_goal_anisotropies,
            params.n_growth_steps,
            jax_arrays,
            params,
        )
        io_utils.save_pkl(cache_path, best_growth_evolution)

    return best_growth_evolution


def plot_best_growth(growth_evolution, output, param_string, params):
    figure = plotting.MorphGrowthFigure(params)

    for t, vertices in enumerate(growth_evolution):
        if t % 10 == 0:
            fig_path = output.file_path(
                f"{BEST_GROWTH_DIR}", param_string, f"step={t:03d}.png"
            )
            figure.save_plot(vertices, t, fig_path)
    fig_path = output.file_path(
        f"{BEST_GROWTH_DIR}", param_string, f"step={t:03d}.png"
    )
    figure.save_plot(vertices, t, fig_path)
