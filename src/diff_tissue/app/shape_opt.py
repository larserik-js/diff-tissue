from ..core import morphing as morphing_core
from ..core import shape_opt as shape_opt_core
from . import config, io_utils, parameters, plotting


class ShapeOptPaths(config.ProjectPaths):
    def __init__(self, base_paths):
        super().__init__(
            data_base_dir=base_paths.data_base_dir,
            outputs_base_dir=base_paths.outputs_base_dir,
        )
        self.final_tissues_dir = self.outputs_base_dir / "final_tissues"
        self.best_morph_data_dir = self.processed_data_dir / "best_morph"
        self.best_morph_figs_dir = self.outputs_base_dir / "best_morph"


def plot_final_tissues(final_tissues, params, output_dir):
    figure = plotting.MorphFigure(params)

    for t, vertices in enumerate(final_tissues):
        if t % 10 == 0 or t == len(final_tissues) - 1:
            figure.update(vertices, enumerate=True)
            fig_path = output_dir / f"step={t:03d}.png"
            io_utils.save_pdf(fig_path, figure.fig, dpi=100)


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
        best_morph_evolution = morphing_core.iterate(
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
        if t % 10 == 0 or t == len(morph_evolution) - 1:
            figure.update(vertices, t)
            fig_path = output_dir / f"step={t:03d}.png"
            io_utils.save_pdf(fig_path, figure.fig, dpi=100)
