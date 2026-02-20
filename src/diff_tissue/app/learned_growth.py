from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.jax_bootstrap import jnp
from ..core import morphing, my_utils
from . import io_utils, parameters, plotting


OUTPUT_TYPE_DIR = "learned_growth"


def _assign_weighted_goals(old_polygons, goals, new_polygons):
    new_goals = []

    for new_poly in new_polygons:
        inter_areas = []
        goals_ = []

        for old_poly, goal in zip(old_polygons, goals):
            inter = new_poly.intersection(old_poly)
            if not inter.is_empty:
                inter_areas.append(inter.area)
                goals_.append(goal)

        goals_ = np.array(goals_)
        inter_areas = np.array(inter_areas)
        total_inter_area = inter_areas.sum()
        weights = inter_areas / total_inter_area

        if np.isclose(total_inter_area, 0.0):
            new_goals.append(0.0)
        else:
            new_goal = np.sum(weights * goals_)
            new_goals.append(new_goal)

    new_goals = jnp.array(new_goals)

    return new_goals


def plot(results, params, output):
    param_string = parameters.get_param_string(params)
    figure = plotting.MorphFigure(results.new_jax_arrays, results.new_params)
    for t, vertices in enumerate(results.growth_evolution):
        if t % 10 == 0:
            fig_path = output.file_path(param_string, f"step={t:03d}.png")
            figure.save_plot(vertices, fig_path)
    fig_path = output.file_path(param_string, f"step={t:03d}.png")
    figure.save_plot(vertices, fig_path)


@dataclass
class _Results:
    growth_evolution: jnp.ndarray
    new_jax_arrays: dict
    new_params: parameters.Params


def run(params, output):
    jax_arrays = my_utils.get_jax_arrays(params)

    old_polygons = my_utils.get_shapely_polygons(
        jax_arrays["init_vertices"], jax_arrays["indices"]
    )

    input_file = io_utils.get_output_params_file(params)
    df = pd.read_csv(input_file, sep="\t", index_col=0)

    goal_areas = df["best_goal_area"].values
    goal_anisotropies = df["best_goal_anisotropy"].values

    # Regenerate new system
    new_params = params.replace(seed=params.seed)
    new_jax_arrays = my_utils.get_jax_arrays(new_params)
    new_polygons = my_utils.get_shapely_polygons(
        new_jax_arrays["init_vertices"], new_jax_arrays["indices"]
    )

    resulting_areas = _assign_weighted_goals(
        old_polygons, goal_areas, new_polygons
    )
    resulting_anisotropies = _assign_weighted_goals(
        old_polygons, goal_anisotropies, new_polygons
    )

    growth_evolution = morphing.iterate(
        resulting_areas,
        resulting_anisotropies,
        new_params.n_growth_steps,
        jax_arrays,
        new_params,
    )

    param_string = parameters.get_param_string(params)
    output_path = output.cache_path(f"{param_string}.pkl")
    io_utils.save_pkl(output_path, growth_evolution)

    results = _Results(
        growth_evolution=growth_evolution,
        new_jax_arrays=new_jax_arrays,
        new_params=new_params,
    )

    return results
