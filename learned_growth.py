import jax.numpy as jnp
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union


import growth, my_utils


def _make_poly_idx_lists(jax_arrays):
    polygon_indices = jax_arrays['indices']
    poly_idx_lists = []

    for polygon in polygon_indices:
        poly_inds = polygon[polygon != -1]
        poly_idx_list = poly_inds[:-2]
        poly_idx_lists.append(poly_idx_list)
    return poly_idx_lists


def build_polygons(jax_arrays):
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


def assign_weighted_goal_areas(old_polygons, goal_areas, new_polygons):
    new_goal_areas = []

    for new_poly in new_polygons:
        intersections = []
        goal_parts = []
        goal_areas_ = []

        for old_poly, goal_area in zip(old_polygons, goal_areas):
            inter = new_poly.intersection(old_poly)
            if not inter.is_empty:
                inter_area = inter.area
                intersections.append(inter_area)
                goal_parts.append((inter_area, goal_area))
                goal_areas_.append(goal_area)
        
        intersections = np.array(intersections)
        goal_areas_ = np.array(goal_areas_)
        total_inter_area = intersections.sum()

        if np.isclose(total_inter_area, 0.0):
            new_goal_areas.append(0.0)
        else:
            # weighted_sum = sum((inter_area / total_inter_area) * goal_area
            #                    for inter_area, goal_area in goal_parts)
            weighted_goal_area = np.sum(
                intersections * goal_areas_ / total_inter_area
            )
            # new_goal_areas.append(weighted_sum)
            new_goal_areas.append(weighted_goal_area)
    
    new_goal_areas = np.array(new_goal_areas)

    return new_goal_areas


def main():
    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    learned_growth_dir = my_utils.OutputDir('learned_growth', params)
    learned_growth_dir.make()

    arrays = my_utils.get_arrays(params)
    old_polygons = build_polygons(arrays)

    input_file = my_utils.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)
    
    goal_areas = df['goal_area'].values
    # goal_aspect_ratios = df['goal_aspect_ratio'].values

    # Regenerate new system
    np.random.seed(10001)
    new_arrays = my_utils.get_arrays(params)
    new_polygons = build_polygons(new_arrays)

    resulting_areas = assign_weighted_goal_areas(
        old_polygons, goal_areas, new_polygons
    )

    resulting_areas = jnp.array(resulting_areas)
    mock_aspect_ratios = 0.5 * jnp.ones_like(resulting_areas)
    
    jax_arrays = my_utils._make_jax_arrays(new_arrays)

    growth.iterate_and_plot(
        learned_growth_dir.get_param_path(), resulting_areas,
        mock_aspect_ratios, jax_arrays, params.numerical
    )


if __name__ == '__main__':
    main()
