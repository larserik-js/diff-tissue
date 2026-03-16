from ..core import shape_opt as shape_opt_core
from . import io_utils, morphing, parameters, plotting


OUTPUT_TYPE_DIR = "shape_opt"
FINAL_TISSUES_DIR = "final_tissues"
BEST_MORPH_DIR = "best_morph"


def plot_final_tissues(final_tissues, output, params):
    param_string = parameters.get_param_string(params)
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


def get_sim_states(params, output):
    param_string = parameters.get_param_string(params)
    cache_path = output.cache_path(f"sim_states__{param_string}.pkl")

    if cache_path.exists():
        sim_states = io_utils.load_pkl(cache_path)
    else:
        sim_states = shape_opt_core.run(params)
        io_utils.save_pkl(cache_path, sim_states)

    return sim_states


def get_best_morph_evolution(
    best_goal_areas, best_goal_anisotropies, polygons, params, cache_path
):
    if cache_path.exists():
        best_morph_evolution = io_utils.load_pkl(cache_path)
    else:
        best_morph_evolution = morphing.jiterate(
            best_goal_areas,
            best_goal_anisotropies,
            params.n_morph_steps,
            polygons,
            params,
        )
        io_utils.save_pkl(cache_path, best_morph_evolution)

    return best_morph_evolution


def plot_best_morph(morph_evolution, output, param_string, params):
    figure = plotting.MorphGrowthFigure(params)

    for t, vertices in enumerate(morph_evolution):
        if t % 10 == 0:
            fig_path = output.file_path(
                f"{BEST_MORPH_DIR}", param_string, f"step={t:03d}.png"
            )
            figure.save_plot(vertices, t, fig_path)
    fig_path = output.file_path(
        f"{BEST_MORPH_DIR}", param_string, f"step={t:03d}.png"
    )
    figure.save_plot(vertices, t, fig_path)
