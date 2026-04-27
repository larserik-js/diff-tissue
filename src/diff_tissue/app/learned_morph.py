from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.jax_bootstrap import jnp
from ..core import shape_opt as shape_opt_core
from ..core import init_systems, morphing
from . import shape_opt as shape_opt_app
from . import io_utils, parameters, plotting


class LearnedMorphPaths:
    def __init__(self, project_paths, param_string):
        self._project_paths = project_paths
        self._param_string = param_string
        self._output_type = "learned_morph"

    @property
    def sim_states_data_path(self):
        sim_states_data_path_ = shape_opt_app.ShapeOptPaths(
            self._project_paths, self._param_string
        ).sim_states_data_path
        return sim_states_data_path_

    @property
    def _data_dir(self):
        data_dir_ = Path(
            self._project_paths.processed_data_dir, self._output_type
        )
        return data_dir_

    @property
    def data_output_path(self):
        output_path_ = Path(self._data_dir, f"{self._param_string}.npz")
        return output_path_

    @property
    def figs_dir(self):
        figs_dir_ = Path(
            self._project_paths.outputs_base_dir,
            self._output_type,
            self._param_string,
        )
        return figs_dir_


def _assign_weighted_goals(old_polygons, goals, new_polygons):
    new_goals_list = []

    for new_poly in new_polygons:
        inter_areas_list = []
        goals_list = []

        for old_poly, goal in zip(old_polygons, goals):
            inter = new_poly.intersection(old_poly)
            if not inter.is_empty:
                inter_areas_list.append(inter.area)
                goals_list.append(goal)

        goals_ = np.array(goals_list)
        inter_areas = np.array(inter_areas_list)
        total_inter_area = inter_areas.sum()
        weights = inter_areas / total_inter_area

        if np.isclose(total_inter_area, 0.0):
            new_goals_list.append(0.0)
        else:
            new_goal = np.sum(weights * goals_)
            new_goals_list.append(new_goal)

    new_goals = jnp.array(new_goals_list)

    return new_goals


def plot(results, output_dir):
    figure = plotting.MorphFigure(results.params)
    io_utils.ensure_dir(output_dir)
    for t, vertices in enumerate(results.morph_evolution):
        if t % 10 == 0 or t == len(results.morph_evolution) - 1:
            figure.update(vertices)
            fig_path = output_dir / f"step={t:03d}.png"
            io_utils.save_pdf(fig_path, figure.fig, dpi=100)


@dataclass
class _Results:
    morph_evolution: np.ndarray
    params: parameters.Params


def run(params, learned_morph_paths):
    input_seed = params.seed  # Store for regenerated system
    replaced_params = params.replace(
        seed=0
    )  # Always base on same initial system

    polygons = init_systems.get_system(replaced_params)
    old_shapely_polygons = init_systems.get_shapely_polygons(
        polygons.init_vertices, polygons.indices
    )

    sim_states = shape_opt_app.get_sim_states(
        replaced_params, learned_morph_paths.sim_states_data_path
    )
    best_state = shape_opt_core.get_best_state(sim_states)
    goal_areas = best_state.goal_areas
    goal_anisotropies = best_state.goal_anisotropies

    # Regenerate new system, base on cli seed
    params = replaced_params.replace(seed=input_seed)
    new_polygons = init_systems.get_jax_polygons(params)

    new_shapely_polygons = init_systems.get_shapely_polygons(
        new_polygons.init_vertices, new_polygons.indices
    )

    resulting_areas = _assign_weighted_goals(
        old_shapely_polygons, goal_areas, new_shapely_polygons
    )
    resulting_anisotropies = _assign_weighted_goals(
        old_shapely_polygons, goal_anisotropies, new_shapely_polygons
    )

    morph_evolution = morphing.iterate(
        resulting_areas,
        resulting_anisotropies,
        params.n_morph_steps,
        new_polygons,
        params,
    )

    morph_evolution_np = np.array(morph_evolution)

    io_utils.ensure_parent_dir(learned_morph_paths.data_output_path)
    io_utils.save_arrays(
        learned_morph_paths.data_output_path,
        morph_evolution=morph_evolution_np,
    )

    results = _Results(
        morph_evolution=morph_evolution_np,
        params=params,
    )

    return results
