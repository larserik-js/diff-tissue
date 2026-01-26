from dataclasses import dataclass
import pathlib
import pickle

import jax
import numpy as np
import shapely
from shapely.strtree import STRtree

from diff_tissue import my_utils


@dataclass
class _Mesh:
    polygons: list
    areas: np.ndarray
    elongations: np.ndarray


def _get_outer_shape():
    np.random.seed(0)
    params = my_utils.Params()
    jax_arrays = my_utils.get_jax_arrays(params)
    outer_shape = jax_arrays['outer_shape']
    return outer_shape


def _make_samples(nx, ny, outer_shape):
    xmin, ymin = outer_shape.min(axis=0)
    xmax, ymax = outer_shape.max(axis=0)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    all_points = np.column_stack([X.ravel(), Y.ravel()])

    return all_points


def _get_inside_shape_mask(outer_shape, sample_coords):
    domain_polygon = shapely.Polygon(outer_shape)
    sample_coords_shapely = shapely.points(sample_coords)
    inside_shape_mask = domain_polygon.covers(sample_coords_shapely)
    return inside_shape_mask


def _get_mapped_metrics(jax_arrays):
    all_mapped_cells = my_utils.get_all_cells(
        jax_arrays['mapped_vertices'], jax_arrays['indices']
    )
    mapped_areas = my_utils.calc_all_areas(
        all_mapped_cells, jax_arrays['valid_mask']
    )
    mapped_elongations = my_utils.calc_elongations(
        all_mapped_cells, jax_arrays['valid_mask']
    )
    return mapped_areas, mapped_elongations


def _build_meshes(n_meshes, output_file):
    if output_file.exists():
        with open(output_file, 'rb') as f:
            meshes = pickle.load(f)
        return meshes
    else:
        params = my_utils.Params()
        meshes = []

        print('Building meshes...')
        for i in range(n_meshes):
            if (i+1)%10 == 0:
                print(f'{i+1} / {n_meshes}')
            params.numerical['seed'] = i
            np.random.seed(params.numerical['seed'])
            jax_arrays = my_utils.get_jax_arrays(params)
            shapely_polygons = my_utils.get_shapely_polygons(
                jax_arrays['mapped_vertices'], jax_arrays['indices']
            )
            mapped_areas, mapped_elongations = _get_mapped_metrics(jax_arrays)

            mesh = _Mesh(
                shapely_polygons, np.array(mapped_areas),
                np.array(mapped_elongations)
            )
            meshes.append(mesh)
    
        with open(output_file, 'wb') as f:
            pickle.dump(meshes, f)
    
    return meshes


def _sample_mesh(mesh: _Mesh, points_inside_shape: np.ndarray):
    """
    Assign mesh scalar values to sample points.
    Points must already be NumPy and lie inside the domain.
    """
    # predicate must be 'intersects'
    # index order is (point_index, polygon_index)
    tree = STRtree(mesh.polygons)
    points_shapely = shapely.points(points_inside_shape)
    matches = tree.query(points_shapely, predicate='intersects')
    point_inds, poly_inds = matches

    sampled_areas = np.full(len(points_inside_shape), np.nan)
    sampled_elongations = np.full(len(points_inside_shape), np.nan)

    sampled_areas[point_inds] = mesh.areas[poly_inds]
    sampled_elongations[point_inds] = mesh.elongations[poly_inds]

    return sampled_areas, sampled_elongations


def _calc_mean_metrics(all_sampled_metrics: list):
    stacked_metrics = np.vstack(all_sampled_metrics)
    mean_metrics = np.nanmean(stacked_metrics, axis=0)
    return mean_metrics


def _get_mean_mapped_fields(meshes, points_inside_shape):
    """
    Sample all meshes and average their scalar fields.
    """
    all_sampled_areas = []
    all_sampled_elongations = []

    for mesh in meshes:
        sampled_areas, sampled_elongations = _sample_mesh(
            mesh, points_inside_shape
        )
        all_sampled_areas.append(sampled_areas)
        all_sampled_elongations.append(sampled_elongations)

    mean_areas = _calc_mean_metrics(all_sampled_areas)
    mean_elongations = _calc_mean_metrics(all_sampled_elongations)

    return mean_areas, mean_elongations


def _save_mapped_fields(coords, mapped_area_field, mapped_elongation_field,
                        output_file):
    output = {
        'coords': coords,
        'mapped_area_field': mapped_area_field,
        'mapped_elongation_field': mapped_elongation_field
    }
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)


def _main():
    jax.config.update('jax_enable_x64', True)

    outer_shape = _get_outer_shape()

    nx, ny = 100, 100
    sample_coords = _make_samples(nx, ny, outer_shape)

    inside_shape_mask = _get_inside_shape_mask(outer_shape, sample_coords)
    points_inside_shape = sample_coords[inside_shape_mask]

    output_dir = pathlib.Path('outputs')
    output_dir.mkdir(exist_ok=True)

    meshes_file = output_dir / 'meshes.pkl'
    meshes = _build_meshes(n_meshes=100, output_file=meshes_file)

    mapped_area_field, mapped_elongation_field = _get_mean_mapped_fields(
        meshes, points_inside_shape
    )

    mapped_fields_file = output_dir / 'mapped_fields.pkl'
    _save_mapped_fields(
        points_inside_shape, mapped_area_field, mapped_elongation_field,
        mapped_fields_file
    )


if __name__ == '__main__':
    _main()
