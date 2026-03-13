from dataclasses import dataclass

import numpy as np

from ..core.jax_bootstrap import jnp
from ..core import shape_opt as shape_opt_core
from ..core import init_systems, morphing, my_utils
from . import shape_opt as shape_opt_app
from . import io_utils, parameters, plotting


OUTPUT_TYPE_DIR = "learned_growth"


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


def plot(results, params, output):
    param_string = parameters.get_param_string(params)
    figure = plotting.MorphFigure(results.new_params)
    for t, vertices in enumerate(results.growth_evolution):
        if t % 10 == 0:
            fig_path = output.file_path(param_string, f"step={t:03d}.png")
            figure.save_plot(vertices, fig_path)
    fig_path = output.file_path(param_string, f"step={t:03d}.png")
    figure.save_plot(vertices, fig_path)


@dataclass
class _Results:
    growth_evolution: jnp.ndarray
    new_params: parameters.Params


def run(params, output):
    input_seed = params.seed  # Store for regenerated system
    params = params.replace(seed=0)  # Always base on same initial system

    polygons = init_systems.get_system(params)
    old_shapely_polygons = my_utils.get_shapely_polygons(
        polygons.init_vertices, polygons.indices
    )

    sim_states = shape_opt_app.get_sim_states(params, output)
    best_state = shape_opt_core.get_best_state(sim_states)
    goal_areas = best_state.goal_areas
    goal_anisotropies = best_state.goal_anisotropies

    # Regenerate new system, base on cli seed
    new_params = params.replace(seed=input_seed)
    new_polygons = init_systems.get_jax_polygons(new_params)

    new_shapely_polygons = my_utils.get_shapely_polygons(
        new_polygons.init_vertices, new_polygons.indices
    )

    resulting_areas = _assign_weighted_goals(
        old_shapely_polygons, goal_areas, new_shapely_polygons
    )
    resulting_anisotropies = _assign_weighted_goals(
        old_shapely_polygons, goal_anisotropies, new_shapely_polygons
    )

    growth_evolution = morphing.iterate(
        resulting_areas,
        resulting_anisotropies,
        new_params.n_growth_steps,
        new_polygons,
        new_params,
    )

    param_string = parameters.get_param_string(new_params)
    output_path = output.cache_path(f"{param_string}.pkl")
    io_utils.save_pkl(output_path, growth_evolution)

    results = _Results(
        growth_evolution=growth_evolution,
        new_params=new_params,
    )

    return results
