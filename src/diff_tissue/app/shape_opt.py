from ..core import shape_opt as shape_opt_core
from . import io_utils, morphing, parameters, plotting


OUTPUT_TYPE_DIR = "shape_opt"
FINAL_TISSUES_DIR = "final_tissues"
BEST_MORPH_DIR = "best_morph"


def plot_final_tissues(final_tissues, params, output_dir):
    figure = plotting.MorphFigure(params)

    for t, vertices in enumerate(final_tissues):
        if t % 10 == 0:
            fig_path = output_dir / f"step={t:03d}.png"
            figure.save_plot(vertices, fig_path, enumerate=True)
    fig_path = output_dir / f"step={t:03d}.png"
    figure.save_plot(vertices, fig_path, enumerate=True)


def get_sim_states(params, paths):
    param_string = parameters.get_param_string(params)
    data_path = paths.sim_states_dir / f"{param_string}.pkl"

    if data_path.exists():
        sim_states = io_utils.load_pkl(data_path)
    else:
        sim_states = shape_opt_core.run(params)
        io_utils.save_pkl(data_path, sim_states)

    return sim_states


def get_best_morph_evolution(
    best_goal_areas, best_goal_anisotropies, polygons, params, data_path
):
    if data_path.exists():
        best_morph_evolution = io_utils.load_pkl(data_path)
    else:
        best_morph_evolution = morphing.jiterate(
            best_goal_areas,
            best_goal_anisotropies,
            params.n_morph_steps,
            polygons,
            params,
        )
        io_utils.save_pkl(data_path, best_morph_evolution)

    return best_morph_evolution


def plot_best_morph(morph_evolution, params, output_dir):
    figure = plotting.MorphGrowthFigure(params)

    for t, vertices in enumerate(morph_evolution):
        if t % 10 == 0:
            fig_path = output_dir / f"step={t:03d}.png"
            figure.save_plot(vertices, t, fig_path)
    fig_path = output_dir / f"step={t:03d}.png"
    figure.save_plot(vertices, t, fig_path)
