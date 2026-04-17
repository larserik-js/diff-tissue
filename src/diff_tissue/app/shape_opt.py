from dataclasses import fields
from itertools import product
import multiprocessing as mp

from ..core import morphing as morphing_core
from ..core import shape_opt as shape_opt_core
from . import io_utils, parameters, plotting


class ShapeOptPaths:
    def __init__(self, project_paths, param_string):
        self._project_paths = project_paths
        self._param_string = param_string
        self._best_morph_name = "best_morph"

    @property
    def final_tissues_dir(self):
        final_tissues_dir_ = self._project_paths.make_subdir(
            self._project_paths.outputs_base_dir,
            "final_tissues",
            self._param_string,
        )
        return final_tissues_dir_

    @property
    def _best_morph_data_dir(self):
        best_morph_data_dir_ = self._project_paths.make_subdir(
            self._project_paths.processed_data_dir / self._best_morph_name
        )
        return best_morph_data_dir_

    @property
    def best_morph_data_path(self):
        best_morph_data_path_ = (
            self._best_morph_data_dir / f"{self._param_string}.pkl"
        )
        return best_morph_data_path_

    @property
    def best_morph_figs_dir(self):
        best_morph_figs_dir_ = self._project_paths.make_subdir(
            self._project_paths.outputs_base_dir / self._best_morph_name,
            self._param_string,
        )
        return best_morph_figs_dir_


def plot_final_tissues(final_tissues, params, output_dir):
    figure = plotting.MorphFigure(params)

    for t, vertices in enumerate(final_tissues):
        if t % 10 == 0 or t == len(final_tissues) - 1:
            figure.update(vertices, enumerate=True)
            fig_path = output_dir / f"step={t:03d}.png"
            io_utils.save_pdf(fig_path, figure.fig, dpi=100)


def get_sim_states(params, paths):
    sim_states_dir = paths.make_subdir(paths.processed_data_dir / "sim_states")
    param_string = parameters.get_param_string(params)
    data_path = sim_states_dir / f"{param_string}.pkl"

    if data_path.exists():
        sim_states = io_utils.load_pkl(data_path)
    else:
        sim_states = shape_opt_core.run(params)
        io_utils.save_pkl(data_path, sim_states)

    return sim_states


def _grid_vars_to_param_combs(grid_vars):
    field_names = [f.name for f in fields(grid_vars)]
    grid_values = [getattr(grid_vars, name) for name in field_names]

    all_param_combs = []
    for values in product(*grid_values):
        combo = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in zip(field_names, values)
        }
        all_param_combs.append(parameters.Params(**combo))
    return all_param_combs


def run_multi(grid_variables, paths, n_workers):
    all_param_combs = _grid_vars_to_param_combs(grid_variables)

    if len(all_param_combs) == 1:
        params = all_param_combs[0]
        params = params.replace(quiet=False)
        _ = get_sim_states(params, paths)

    else:
        inputs = [(param_comb, paths) for param_comb in all_param_combs]

        results = []
        with mp.Pool(processes=n_workers) as pool:
            print(
                f"Running {len(all_param_combs)} trials with "
                f"{n_workers} workers..."
            )
            for result in pool.starmap(get_sim_states, inputs):
                results.append(result)

        print("All trials completed.")
        print("")


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
