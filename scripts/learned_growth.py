import numpy as np
import pandas as pd

from diff_tissue.jax_bootstrap import jnp
from diff_tissue import morphing, io_utils, my_utils, parameters, plotting


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


def _save_growth_evolution(growth_evolution, params):
    output_file = io_utils.OutputFile('learned_growth', '.pkl', params)
    data_handler = io_utils.DataHandler(output_file)
    data_handler.save(growth_evolution)


def _plot(growth_evolution, output_dir, jax_arrays, params):
    figure = plotting.MorphFigure(output_dir, jax_arrays, params)

    for t, vertices in enumerate(growth_evolution):
        if t%10 == 0:
            figure.save_plot(vertices, t)
    figure.save_plot(vertices, t)


def _main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    old_polygons = my_utils.get_shapely_polygons(
        jax_arrays['init_vertices'], jax_arrays['indices']
    )

    input_file = io_utils.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)
    
    goal_areas = df['best_goal_area'].values
    goal_anisotropies = df['best_goal_anisotropy'].values

    # Regenerate new system
    params = params.replace(seed=10000)
    new_arrays = my_utils.get_jax_arrays(params)
    new_polygons = my_utils.get_shapely_polygons(
        new_arrays['init_vertices'], new_arrays['indices']
    )

    resulting_areas = _assign_weighted_goals(
        old_polygons, goal_areas, new_polygons
    )
    resulting_anisotropies = _assign_weighted_goals(
        old_polygons, goal_anisotropies, new_polygons
    )

    jax_arrays = my_utils._make_jax_arrays(new_arrays)

    growth_evolution = morphing.iterate(
        resulting_areas, resulting_anisotropies, params.n_growth_steps,
        jax_arrays, params
    )

    _save_growth_evolution(growth_evolution, params)

    output_dir = io_utils.OutputDir('learned_growth', params).path
    _plot(growth_evolution, output_dir, jax_arrays, params)


if __name__ == '__main__':
    _main()
