from functools import cached_property
import timeit

import numpy as np
from shapely.geometry import Polygon

from .jax_bootstrap import jax, jnp, struct
from . import init_systems, shapes, tutte


def timer(func):
    def timed(*args, **kwargs):
        t_init = timeit.default_timer()
        res = func(*args, **kwargs)
        t_end = timeit.default_timer()

        t_tot = t_end - t_init

        print(f"Total time: {t_tot:.4f} s")
        return res

    return timed


def get_all_cells(vertices, indices):
    all_cells = vertices[indices]
    return all_cells


def calc_centroids(vertices, indices, valid_mask):
    polygons = vertices[indices]
    mask = valid_mask[..., None].repeat(2, axis=2)
    polygons = np.where(mask, polygons, jnp.nan)
    centroids = np.nanmean(polygons, axis=1)
    return centroids


def calc_areas(all_cells, valid_mask):
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


def calc_anisotropies(all_cells, valid_mask):
    xs = all_cells[:, 1:-1, 0]
    ys = all_cells[:, 1:-1, 1]
    valid = valid_mask[:, 1:-1]

    xs_masked = jnp.asarray(jnp.where(valid, xs, jnp.nan))
    ys_masked = jnp.asarray(jnp.where(valid, ys, jnp.nan))

    x_vars = jnp.nanvar(xs_masked, axis=1)
    y_vars = jnp.nanvar(ys_masked, axis=1)

    eps = 1e-8
    anisotropies = (y_vars - x_vars) / (y_vars + x_vars + eps)

    return anisotropies


def calc_masked_cosines(all_cells, valid_mask):
    edges = all_cells[:, 1:] - all_cells[:, :-1]
    epsilon = 1e-6
    norms = jnp.linalg.norm(edges + epsilon, axis=2)
    dot_products = jnp.sum(edges[:, :-1] * edges[:, 1:], axis=2)

    cosines = dot_products / (epsilon + norms[:, :-1] * norms[:, 1:])
    clip_value = 1.0 - epsilon
    cosines = jnp.clip(cosines, -clip_value, clip_value)

    valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    valid = valid[:, 1:] & valid[:, :-1]
    masked_cosines = jnp.where(valid, cosines, jnp.nan)

    return masked_cosines


def calc_optimal_angles(valid_mask):
    n_vertices = valid_mask.sum(axis=1) - 2
    interior_angles = (n_vertices - 2) * jnp.pi / n_vertices
    optimal_angles = jnp.pi - interior_angles
    optimal_angles = optimal_angles[:, None]
    return optimal_angles


@struct.dataclass
class PolyMetrics:
    _indices: jnp.ndarray
    _valid_mask: jnp.ndarray
    areas: jnp.ndarray
    anisotropies: jnp.ndarray
    masked_cosines: jnp.ndarray


def _calc_poly_metrics(vertices, indices, valid_mask):
    all_cells = get_all_cells(vertices, indices)

    areas = calc_areas(all_cells, valid_mask)
    anisotropies = calc_anisotropies(all_cells, valid_mask)
    masked_cosines = calc_masked_cosines(all_cells, valid_mask)

    return areas, anisotropies, masked_cosines


def initialize_poly_metrics(vertices, indices, valid_mask):
    areas, anisotropies, masked_cosines = _calc_poly_metrics(
        vertices, indices, valid_mask
    )

    return PolyMetrics(
        _indices=indices,
        _valid_mask=valid_mask,
        areas=areas,
        anisotropies=anisotropies,
        masked_cosines=masked_cosines,
    )


def update_poly_metrics(poly_metrics, vertices):
    areas, anisotropies, masked_cosines = _calc_poly_metrics(
        vertices, poly_metrics._indices, poly_metrics._valid_mask
    )

    poly_metrics = poly_metrics.replace(
        areas=areas,
        anisotropies=anisotropies,
        masked_cosines=masked_cosines,
    )
    return poly_metrics


class TutteMetrics:
    def __init__(self, polygons, shape):
        self._polygons = polygons
        self._shape = shape

    @cached_property
    def vertices(self):
        target_boundary = shapes.get_target_boundary(
            self._shape,
            self._polygons.mesh_area,
            init_systems.VertexNumbers(self._polygons),
        )
        vertices_ = tutte.get_mapped_vertices(
            self._polygons.vertices,
            self._polygons.polygon_inds,
            self._polygons.boundary_inds,
            target_boundary.vertices,
        )
        return vertices_

    @cached_property
    def _all_cells(self):
        all_cells = get_all_cells(self.vertices, self._polygons.polygon_inds)
        return all_cells

    @cached_property
    def centroids(self):
        centroids_ = calc_centroids(
            self.vertices,
            self._polygons.polygon_inds,
            self._polygons.valid_mask,
        )
        return centroids_

    @cached_property
    def areas(self):
        areas_ = calc_areas(self._all_cells, self._polygons.valid_mask)
        return areas_

    @cached_property
    def anisotropies(self):
        anisotropies_ = calc_anisotropies(
            self._all_cells, self._polygons.valid_mask
        )
        return anisotropies_


def get_tutte_metrics(params):
    polygons = init_systems.get_system(params)
    tutte_metrics = TutteMetrics(polygons, params.shape)
    return tutte_metrics


def _make_array_dict(polygons, target_boundary):
    arrays = {
        "indices": polygons.polygon_inds,
        "valid_mask": polygons.valid_mask,
        "init_vertices": polygons.vertices,
        "poly_neighbors": polygons.poly_neighbors,
        "vertex_neighbors": polygons.vertex_neighbors,
        "vertex_polygons": polygons.vertex_polygons,
        "free_mask": polygons.free_mask,
        "boundary_inds": polygons.boundary_inds,
        "target_boundary": target_boundary.vertices,
        "target_boundary_segments": target_boundary.segments,
    }
    return arrays


def get_arrays(params):
    polygons = init_systems.get_system(params)

    vertex_numbers = init_systems.VertexNumbers(polygons)
    target_boundary = shapes.get_target_boundary(
        params.shape, polygons.mesh_area, vertex_numbers
    )

    arrays = _make_array_dict(
        polygons,
        target_boundary,
    )
    return arrays


def _get_device():
    return jax.devices("cpu")[0]


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
