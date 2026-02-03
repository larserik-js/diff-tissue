from functools import cached_property
import timeit

import numpy as np
from shapely.geometry import Polygon

from .jax_bootstrap import jax, jnp
from . import diffeomorphism, init_systems, shapes


def timer(func):
    def timed(*args, **kwargs):
        t_init = timeit.default_timer()
        res = func(*args, **kwargs)
        t_end = timeit.default_timer()

        t_tot = t_end - t_init

        print(f'Total time: {t_tot:.4f} s')
        return res

    return timed


def calc_centroids(vertices, indices, valid_mask):
    polygons = vertices[indices]
    mask = valid_mask[..., None].repeat(2, axis=2)
    polygons = np.where(mask, polygons, jnp.nan)
    centroids = np.nanmean(polygons, axis=1)
    return centroids


class MappedMetrics:
    def __init__(self, polygons, shape):
        self._polygons = polygons
        self._shape = shape

    @cached_property
    def vertices(self):
        outer_shape = shapes.get_outer_shape(
            self._shape, self._polygons.mesh_area,
            init_systems.VertexNumbers(self._polygons)
        )
        vertices_ = diffeomorphism.get_mapped_vertices(
            self._polygons.vertices, self._polygons.polygon_inds,
            self._polygons.boundary_mask, outer_shape
        )
        return vertices_

    @cached_property
    def centroids(self):
        centroids_ = calc_centroids(
            self.vertices, self._polygons.polygon_inds,
            self._polygons.valid_mask
        )
        return centroids_


def calc_proximal_mask(mapped_centroids, proximal_dist):
    y_dists_from_base = (
        mapped_centroids[:,1] - init_systems.Coords.base_origin[1]
    )
    proximal_mask = (y_dists_from_base <= proximal_dist)
    return proximal_mask


def _make_array_dict(
        polygons, outer_shape, mapped_metrics, proximal_mask, knots
    ):
    arrays = {
        'indices': polygons.polygon_inds,
        'valid_mask': polygons.valid_mask,
        'init_vertices': polygons.vertices,
        'poly_neighbors': polygons.poly_neighbors,
        'vertex_neighbors': polygons.vertex_neighbors,
        'vertex_polygons': polygons.vertex_polygons,
        'free_mask': polygons.free_mask,
        'boundary_mask': polygons.boundary_mask,
        'outer_shape': outer_shape,
        'mapped_vertices': mapped_metrics.vertices,
        'mapped_centroids': mapped_metrics.centroids,
        'proximal_mask': proximal_mask,
        'left_knots': knots.left_knots,
        'center_knots': knots.center_knots,
        'right_knots': knots.right_knots,
        'all_knots': knots.all_knots
    }
    return arrays


def get_arrays(params):
    polygons = init_systems.get_system(params.system, params.seed)
    mesh_area = polygons.mesh_area
    vertex_numbers = init_systems.VertexNumbers(polygons)

    outer_shape = shapes.get_outer_shape(
        params.shape, mesh_area, vertex_numbers
    )
    mapped_metrics = MappedMetrics(polygons, params.shape)
    proximal_mask = calc_proximal_mask(
        mapped_metrics.centroids, params.proximal_dist
    )
    knots = init_systems.Knots()
    arrays = _make_array_dict(
        polygons, outer_shape, mapped_metrics, proximal_mask, knots
    )
    return arrays


def _get_device():
    return jax.devices('cpu')[0]


def _send_to_device(jax_array):
    return jax.device_put(jax_array, device=_get_device())


def to_jax(np_array):
    return _send_to_device(jnp.array(np_array))


def _make_jax_arrays(arrays):
    jax_arrays = {name: to_jax(array) for name, array in arrays.items()}
    return jax_arrays


def get_jax_arrays(params):
    arrays = get_arrays(params)
    jax_arrays = _make_jax_arrays(arrays)
    return jax_arrays


def calc_optimal_angles(valid_mask):
    n_vertices = valid_mask.sum(axis=1) - 2
    interior_angles = (n_vertices - 2) * jnp.pi / n_vertices
    optimal_angles = jnp.pi - interior_angles
    optimal_angles = optimal_angles[:, None]
    return optimal_angles


def get_all_cells(vertices, indices):
    all_cells = vertices[indices]
    return all_cells


def calc_all_areas(all_cells, valid_mask):
    xs = all_cells[:, 1:-1, 0]
    y_plus_ones = all_cells[:, 2:, 1]
    y_minus_ones = all_cells[:, :-2, 1]

    valid = valid_mask[:, 1:-1] & valid_mask[:, 2:] & valid_mask[:, :-2]

    first_term = xs * y_plus_ones
    first_term = jnp.sum(first_term * valid, axis=1)
    second_term = xs * y_minus_ones
    second_term = jnp.sum(second_term * valid, axis=1)

    # Assumes vertices are ordered counter-clockwise
    areas = 0.5 * (first_term - second_term)

    return areas


def calc_elongations(all_cells, valid_mask):
    xs = all_cells[:, 1:-1, 0]
    ys = all_cells[:, 1:-1, 1]
    valid = valid_mask[:, 1:-1]

    xs_masked = jnp.where(valid, xs, jnp.nan)
    ys_masked = jnp.where(valid, ys, jnp.nan)

    x_vars = jnp.nanvar(xs_masked, axis=1)
    y_vars = jnp.nanvar(ys_masked, axis=1)

    eps = 1e-8
    elongations = (y_vars - x_vars) / (y_vars + x_vars + eps)

    return elongations


def _make_poly_idx_lists(polygon_indices):
    poly_idx_lists = []

    for polygon in polygon_indices:
        poly_inds = polygon[polygon != -1]
        poly_idx_list = poly_inds[:-2]
        poly_idx_lists.append(poly_idx_list)
    return poly_idx_lists


def get_shapely_polygons(vertices, poly_indices):
    poly_idx_lists = _make_poly_idx_lists(poly_indices)
    polygons = []
    for idx_list in poly_idx_lists:
        coords = vertices[idx_list]
        # Ensure closure: Shapely closes automatically,
        # but doing it explicitly avoids issues
        if not (coords[0] == coords[-1]).all():
            coords = np.vstack([coords, coords[0]])
        polygons.append(Polygon(coords))
    return polygons
