from dataclasses import dataclass
import pickle

import numpy as np
import shapely
from shapely.strtree import STRtree

from ..app import parameters
from . import init_systems, my_utils, shapes


def _get_general_target_boundary(shape):
    general_params = parameters.Params(system="voronoi", seed=0)
    polygons = init_systems.get_system(general_params)
    vertex_numbers = init_systems.VertexNumbers(polygons)
    target_boundary = shapes.get_target_boundary(
        shape, polygons.mesh_area, vertex_numbers
    )
    return target_boundary.vertices


def _make_samples(nx, ny, target_boundary):
    xmin, ymin = target_boundary.min(axis=0)
    xmax, ymax = target_boundary.max(axis=0)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    all_points = np.column_stack([X.ravel(), Y.ravel()])

    return all_points


def _get_inside_shape_mask(target_boundary, sample_coords):
    domain_polygon = shapely.Polygon(target_boundary)
    sample_coords_shapely = shapely.points(sample_coords)
    inside_shape_mask = domain_polygon.covers(sample_coords_shapely)
    return inside_shape_mask


def _get_points_inside_shape(shape, nx, ny):
    target_boundary = _get_general_target_boundary(shape)
    sample_coords = _make_samples(nx, ny, target_boundary)

    inside_shape_mask = _get_inside_shape_mask(target_boundary, sample_coords)
    points_inside_shape = sample_coords[inside_shape_mask]
    return points_inside_shape


@dataclass
class _Mesh:
    polygons: list
    areas: np.ndarray
    anisotropies: np.ndarray


def _build_meshes(shape, n_meshes=100):
    params = parameters.Params()
    params = params.replace(shape=shape)
    meshes = []

    print("Building meshes...")
    for i in range(n_meshes):
        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {n_meshes}")

        params = params.replace(seed=i)
        polygons = init_systems.get_system(params)
        tutte_vertices = my_utils.TutteMetrics(polygons, params.shape).vertices
        shapely_polygons = init_systems.get_shapely_polygons(
            tutte_vertices, polygons.indices
        )
        tutte_metrics = my_utils.TutteMetrics(polygons, shape)

        mesh = _Mesh(
            shapely_polygons,
            np.array(tutte_metrics.areas),
            np.array(tutte_metrics.anisotropies),
        )
        meshes.append(mesh)

    return meshes


def _get_meshes(output_manager, shape):
    meshes_file = output_manager.cache_path(f"meshes__{shape}.pkl")
    if meshes_file.exists():
        with open(meshes_file, "rb") as f:
            meshes = pickle.load(f)
    else:
        meshes = _build_meshes(shape)
        with open(meshes_file, "wb") as f:
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
    matches = tree.query(points_shapely, predicate="intersects")
    point_inds, poly_inds = matches

    sampled_areas = np.full(len(points_inside_shape), np.nan)
    sampled_anisotropies = np.full(len(points_inside_shape), np.nan)

    sampled_areas[point_inds] = mesh.areas[poly_inds]
    sampled_anisotropies[point_inds] = mesh.anisotropies[poly_inds]

    return sampled_areas, sampled_anisotropies


def _calc_mean_metrics(all_sampled_metrics: list):
    stacked_metrics = np.vstack(all_sampled_metrics)
    mean_metrics = np.nanmean(stacked_metrics, axis=0)
    return mean_metrics


def _get_fields(meshes, points_inside_shape):
    """
    Sample all meshes and average their scalar fields.
    """
    all_sampled_areas = []
    all_sampled_anisotropies = []

    for mesh in meshes:
        sampled_areas, sampled_anisotropies = _sample_mesh(
            mesh, points_inside_shape
        )
        all_sampled_areas.append(sampled_areas)
        all_sampled_anisotropies.append(sampled_anisotropies)

    mean_areas = _calc_mean_metrics(all_sampled_areas)
    mean_anisotropies = _calc_mean_metrics(all_sampled_anisotropies)

    return mean_areas, mean_anisotropies


@dataclass
class _TutteFields:
    coords: np.ndarray
    areas: np.ndarray
    anisotropies: np.ndarray


def generate_fields(output_manager, shape):
    points_inside_shape = _get_points_inside_shape(shape, nx=100, ny=100)

    meshes = _get_meshes(output_manager, shape)

    area_field, anisotropy_field = _get_fields(meshes, points_inside_shape)

    tutte_fields_ = _TutteFields(
        points_inside_shape, area_field, anisotropy_field
    )

    return tutte_fields_
