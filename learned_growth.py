import jax.numpy as jnp
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import growth, my_files, my_utils


def _make_poly_idx_lists(jax_arrays):
    polygon_indices = jax_arrays['indices']
    poly_idx_lists = []

    for polygon in polygon_indices:
        poly_inds = polygon[polygon != -1]
        poly_idx_list = poly_inds[:-2]
        poly_idx_lists.append(poly_idx_list)
    return poly_idx_lists


def _build_polygons(jax_arrays):
    poly_idx_lists = _make_poly_idx_lists(jax_arrays)
    polygons = []
    for idx_list in poly_idx_lists:
        coords = jax_arrays['init_vertices'][idx_list]
        # Ensure closure: Shapely closes automatically,
        # but doing it explicitly avoids issues
        if not (coords[0] == coords[-1]).all():
            coords = np.vstack([coords, coords[0]])
        polygons.append(Polygon(coords))
    return polygons


def _assign_weighted_goals(old_polygons, goals, new_polygons):
    new_goals = []

    for new_poly in new_polygons:
        intersections = []
        goal_parts = []
        goals_ = []

        for old_poly, goal in zip(old_polygons, goals):
            inter = new_poly.intersection(old_poly)
            if not inter.is_empty:
                inter_area = inter.area
                intersections.append(inter_area)
                goal_parts.append((inter_area, goal))
                goals_.append(goal)
        
        intersections = np.array(intersections)
        goals_ = np.array(goals_)
        total_inter_area = intersections.sum()

        if np.isclose(total_inter_area, 0.0):
            new_goals.append(0.0)
        else:
            weighted_goal = np.sum(
                intersections * goals_ / total_inter_area
            )
            new_goals.append(weighted_goal)
    
    new_goals = jnp.array(new_goals)

    return new_goals


def _main():
    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    learned_growth_dir = my_files.OutputDir('learned_growth', params)

    arrays = my_utils.get_arrays(params)
    old_polygons = _build_polygons(arrays)

    input_file = my_files.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)
    
    goal_areas = df['goal_area'].values
    goal_aspect_ratios = df['goal_aspect_ratio'].values

    # Regenerate new system
    np.random.seed(10000)
    new_arrays = my_utils.get_arrays(params)
    new_polygons = _build_polygons(new_arrays)

    resulting_areas = _assign_weighted_goals(
        old_polygons, goal_areas, new_polygons
    )
    resulting_aspect_ratios = _assign_weighted_goals(
        old_polygons, goal_aspect_ratios, new_polygons
    )

    jax_arrays = my_utils._make_jax_arrays(new_arrays)

    growth.iterate_and_plot(
        learned_growth_dir.get_path(), resulting_areas,
        resulting_aspect_ratios, jax_arrays, params.numerical
    )


if __name__ == '__main__':
    _main()
