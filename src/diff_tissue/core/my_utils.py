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

    xs_masked = jnp.where(valid, xs, jnp.nan)
    ys_masked = jnp.where(valid, ys, jnp.nan)

    x_vars = jnp.nanvar(xs_masked, axis=1)
    y_vars = jnp.nanvar(ys_masked, axis=1)

    eps = 1e-8
    anisotropies = (y_vars - x_vars) / (y_vars + x_vars + eps)

    return anisotropies


def calc_masked_cosines(all_cells, valid_mask):
    edges = all_cells[:, 1:] - all_cells[:, :-1]
    epsilon = 1e-7
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
    _indices: jnp.ndarray = struct.field(pytree_node=True)
    _valid_mask: jnp.ndarray = struct.field(pytree_node=True)

    areas: jnp.ndarray = struct.field(pytree_node=True)
    anisotropies: jnp.ndarray = struct.field(pytree_node=True)
    masked_cosines: jnp.ndarray = struct.field(pytree_node=True)

    @classmethod
    def create(cls, vertices, indices, valid_mask):
        all_cells = get_all_cells(vertices, indices)

        areas = calc_areas(all_cells, valid_mask)
        anisotropies = calc_anisotropies(all_cells, valid_mask)
        masked_cosines = calc_masked_cosines(all_cells, valid_mask)

        return cls(
            _indices=indices,
            _valid_mask=valid_mask,
            areas=areas,
            anisotropies=anisotropies,
            masked_cosines=masked_cosines,
        )

    def update(self, vertices):
        all_cells = get_all_cells(vertices, self._indices)

        areas = calc_areas(all_cells, self._valid_mask)
        anisotropies = calc_anisotropies(all_cells, self._valid_mask)
        masked_cosines = calc_masked_cosines(all_cells, self._valid_mask)

        return self.replace(
            masked_cosines=masked_cosines,
            areas=areas,
            anisotropies=anisotropies,
        )


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
            self._polygons.boundary_mask,
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


def calc_proximal_mask(tutte_centroids, proximal_dist):
    y_dists_from_base = (
        tutte_centroids[:, 1] - init_systems.Coords.base_origin[1]
    )
    proximal_mask = y_dists_from_base <= proximal_dist
    return proximal_mask


def _make_array_dict(
    polygons, tutte_metrics, target_boundary, proximal_mask, knots
):
    arrays = {
        "indices": polygons.polygon_inds,
        "valid_mask": polygons.valid_mask,
        "init_vertices": polygons.vertices,
        "poly_neighbors": polygons.poly_neighbors,
        "vertex_neighbors": polygons.vertex_neighbors,
        "vertex_polygons": polygons.vertex_polygons,
        "free_mask": polygons.free_mask,
        "boundary_inds": polygons.boundary_inds,
        "boundary_mask": polygons.boundary_mask,
        "tutte_vertices": tutte_metrics.vertices,
        "tutte_centroids": tutte_metrics.centroids,
        "tutte_areas": tutte_metrics.areas,
        "tutte_anisotropies": tutte_metrics.anisotropies,
        "target_boundary": target_boundary.vertices,
        "target_boundary_segments": target_boundary.segments,
        "proximal_mask": proximal_mask,
        "left_knots": knots.left_knots,
        "center_knots": knots.center_knots,
        "right_knots": knots.right_knots,
        "all_knots": knots.all_knots,
    }
    return arrays


def get_arrays(params):
    polygons = init_systems.get_system(params.system, params.seed)

    mesh_area = polygons.mesh_area
    vertex_numbers = init_systems.VertexNumbers(polygons)
    target_boundary = shapes.get_target_boundary(
        params.shape, mesh_area, vertex_numbers
    )

    tutte_metrics = TutteMetrics(polygons, params.shape)
    proximal_mask = calc_proximal_mask(
        tutte_metrics.centroids, params.proximal_dist
    )

    knots = init_systems.Knots()
    arrays = _make_array_dict(
        polygons, tutte_metrics, target_boundary, proximal_mask, knots
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
