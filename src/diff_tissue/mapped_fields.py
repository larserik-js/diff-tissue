from dataclasses import dataclass
import pickle

import numpy as np
import shapely
from shapely.strtree import STRtree

from . import init_systems, my_utils, parameters, shapes


@dataclass
class _Mesh:
    polygons: list
    areas: np.ndarray
    elongations: np.ndarray


def _get_general_outer_shape(shape):
    polygons = init_systems.get_system(system='voronoi', seed=0)
    vertex_numbers = init_systems.VertexNumbers(polygons)
    outer_shape = shapes.get_outer_shape(
        shape, polygons.mesh_area, vertex_numbers
    )
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


def _get_points_inside_shape(shape, nx, ny):
    outer_shape = _get_general_outer_shape(shape)
    sample_coords = _make_samples(nx, ny, outer_shape)

    inside_shape_mask = _get_inside_shape_mask(outer_shape, sample_coords)
    points_inside_shape = sample_coords[inside_shape_mask]
    return points_inside_shape


def _get_mapped_metrics(polygons, mapped_vertices):
    all_mapped_cells = my_utils.get_all_cells(
        mapped_vertices, polygons.polygon_inds
    )
    mapped_areas = my_utils.calc_all_areas(
        all_mapped_cells, polygons.valid_mask
    )
    mapped_elongations = my_utils.calc_elongations(
        all_mapped_cells, polygons.valid_mask
    )
    return mapped_areas, mapped_elongations


def _build_meshes(n_meshes, shape, output_file):
    if output_file.exists():
        with open(output_file, 'rb') as f:
            meshes = pickle.load(f)
        return meshes
    else:
        params = parameters.Params()
        params = params.replace(shape=shape)
        meshes = []

        print('Building meshes...')
        for i in range(n_meshes):
            if (i+1)%10 == 0:
                print(f'{i+1} / {n_meshes}')

            params = params.replace(seed=i)
            polygons = init_systems.get_system(params.system, params.seed)
            mapped_vertices = (
                my_utils.MappedMetrics(polygons, params.shape).vertices
            )
            shapely_polygons = my_utils.get_shapely_polygons(
                mapped_vertices, polygons.polygon_inds
            )
            mapped_areas, mapped_elongations = _get_mapped_metrics(
                polygons, mapped_vertices
            )

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


def run(shape, nx, ny, output_dir):
    points_inside_shape = _get_points_inside_shape(shape, nx, ny)

    meshes_file = output_dir / f'meshes_{shape}.pkl'
    meshes = _build_meshes(n_meshes=100, shape=shape, output_file=meshes_file)

    mapped_area_field, mapped_elongation_field = _get_mean_mapped_fields(
        meshes, points_inside_shape
    )

    mapped_fields_file = output_dir / f'mapped_fields_{shape}.pkl'
    _save_mapped_fields(
        points_inside_shape, mapped_area_field, mapped_elongation_field,
        mapped_fields_file
    )
