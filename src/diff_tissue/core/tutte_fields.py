from dataclasses import dataclass

import numpy as np
import shapely
from shapely.strtree import STRtree

from . import init_systems, metrics, shapes


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


def get_points_inside_shape(target_boundary, nx, ny):
    sample_coords = _make_samples(nx, ny, target_boundary)

    inside_shape_mask = _get_inside_shape_mask(target_boundary, sample_coords)
    points_inside_shape = sample_coords[inside_shape_mask]
    return points_inside_shape


@dataclass
class _Mesh:
    polygons: list
    areas: np.ndarray
    anisotropies: np.ndarray


def build_meshes(params, n_meshes=100):
    meshes = []

    print("Building meshes...")
    for i in range(n_meshes):
        if (i + 1) % 10 == 0:
            print(f"{i + 1} / {n_meshes}")

        params = params.replace(seed=i)
        polygons = init_systems.get_system(params)
        target_boundary = shapes.get_target_boundary(
            params,
            polygons.mesh_area,
            init_systems.VertexNumbers(polygons),
        )
        tutte_metrics = metrics.TutteMetrics(polygons, target_boundary)
        shapely_polygons = init_systems.get_shapely_polygons(
            tutte_metrics.vertices, polygons.indices
        )

        mesh = _Mesh(
            shapely_polygons,
            np.array(tutte_metrics.areas),
            np.array(tutte_metrics.anisotropies),
        )
        meshes.append(mesh)

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


def get_fields(meshes, points_inside_shape):
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
class TutteFields:
    coords: np.ndarray
    areas: np.ndarray
    anisotropies: np.ndarray
