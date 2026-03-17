from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from importlib.resources import files
import json

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Voronoi
import shapely
from shapely.geometry import Polygon

from .jax_bootstrap import jnp, struct


class Coords:
    _origin = np.array([0.0, 0.0])
    base_origin = _origin + 0.0
    full_mesh_origin = np.array([40.0, 0.0])
    full_mesh_base = np.array([40.0, 18.635])


def sort_counterclockwise(indices, vertices):
    centroid = vertices.mean(axis=0)
    angles = np.arctan2(
        vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0]
    )
    inds_for_sorting = np.argsort(angles)
    sorted_poly_inds_arr = np.array(indices)[inds_for_sorting]
    sorted_poly_inds = sorted_poly_inds_arr.tolist()
    return sorted_poly_inds


def get_ccw_boundary_inds(vertices, boundary_inds):
    boundary_vertices = vertices[boundary_inds]

    sorted_boundary_inds = sort_counterclockwise(
        boundary_inds, boundary_vertices
    )
    sorted_boundary_inds = np.array(sorted_boundary_inds)
    return sorted_boundary_inds


def get_segments(vertices):
    closed_polygon = np.concatenate([vertices, vertices[:1]], axis=0)
    segments = closed_polygon[1:] - closed_polygon[:-1]
    return segments


def _extend(indices):
    indices.insert(0, indices[-1])
    indices.append(indices[1])
    return indices


class _Polygons(ABC):
    def __init__(self):
        self._init_vertices: NDArray[np.floating]
        self._indices: NDArray[np.integer]
        self._free_mask: NDArray[np.bool_]
        self._mesh_area: float
        self._build()

        self._valid_mask = self._find_valid_mask()
        self._boundary_inds = self._get_boundary_inds()
        self._poly_neighbors = self._calc_poly_neighbors()
        self._vertex_neighbors = self._calc_vertex_neighbors()
        self._vertex_polygons = self._calc_vertex_polygons()

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def _calc_mesh_area(self):
        pass

    def _find_valid_mask(self):
        valid_mask = self._indices != -1
        return valid_mask

    def _get_boundary_inds(self):
        all_edges = set()
        interior_edges = set()
        for extended_polygon in self._indices:
            unpadded_polygon = extended_polygon[extended_polygon != -1]
            # Removes the last vertex, which was an extension for efficiency
            polygon = unpadded_polygon[:-1]

            for i in range(len(polygon) - 1):
                edge = polygon[i : i + 2]
                sorted_edge = tuple(np.sort(edge))
                if sorted_edge in all_edges:
                    interior_edges.add(sorted_edge)

                all_edges.add(sorted_edge)

        boundary_edges = all_edges - interior_edges
        boundary_inds = np.array(list(boundary_edges)).flatten()
        boundary_inds = np.unique(boundary_inds)
        return boundary_inds

    @staticmethod
    def _list_of_lists_of_ints_to_padded_array(all_neighbors):
        max_n_neighbors = max(
            [len(neighbors_list) for neighbors_list in all_neighbors]
        )
        padded_neighbors = -1 * np.ones(
            (len(all_neighbors), max_n_neighbors), dtype=int
        )
        for i, neighbors in enumerate(all_neighbors):
            n_poly_neighbors = len(neighbors)
            padded_neighbors[i, :n_poly_neighbors] = neighbors

        return padded_neighbors

    def _calc_poly_neighbors(self):
        neighbors = []
        max_n_neighbors = 0

        for idx, vertex_inds_ in enumerate(self._indices):
            vertex_inds = vertex_inds_[vertex_inds_ != -1]
            inds_in_polygons = np.isin(self._indices, vertex_inds)
            poly_neighbors_with_self = np.where(
                np.any(inds_in_polygons, axis=1)
            )[0]
            poly_neighbors = poly_neighbors_with_self[
                poly_neighbors_with_self != idx
            ]
            neighbors.append(poly_neighbors)

            n_neighbors = len(poly_neighbors)
            if n_neighbors > max_n_neighbors:
                max_n_neighbors = n_neighbors

        neighbors = self._list_of_lists_of_ints_to_padded_array(neighbors)
        return neighbors

    def _calc_vertex_neighbors(self):
        vertex_neighbors = defaultdict(set)
        for polygon in self._indices:
            for i in range(len(polygon) - 1):
                vertex_idx, vertex_neighbor_idx = (
                    int(polygon[i]),
                    int(polygon[i + 1]),
                )

                if not vertex_neighbor_idx == -1:
                    vertex_neighbors[vertex_idx].add(int(vertex_neighbor_idx))
                    vertex_neighbors[vertex_neighbor_idx].add(int(vertex_idx))

        vertex_neighbors_lists = [
            list(neighbors_set) for neighbors_set in vertex_neighbors.values()
        ]
        padded_vertex_neighbors = self._list_of_lists_of_ints_to_padded_array(
            vertex_neighbors_lists
        )
        return padded_vertex_neighbors

    def _calc_vertex_polygons(self):
        vertex_polygons_lists = list()
        for vertex_idx in range(self._init_vertices.shape[0]):
            vertex_polygons = np.where(vertex_idx == self._indices)[0]
            vertex_polygons_list = list(set(vertex_polygons))
            vertex_polygons_lists.append(vertex_polygons_list)

        vertex_polygons = self._list_of_lists_of_ints_to_padded_array(
            vertex_polygons_lists
        )
        return vertex_polygons

    @property
    def init_vertices(self):
        return self._init_vertices

    @property
    def indices(self):
        return self._indices

    @property
    def free_mask(self):
        return self._free_mask

    @property
    def valid_mask(self):
        return self._valid_mask

    @property
    def boundary_inds(self):
        return self._boundary_inds

    @property
    def poly_neighbors(self):
        return self._poly_neighbors

    @property
    def vertex_neighbors(self):
        return self._vertex_neighbors

    @property
    def vertex_polygons(self):
        return self._vertex_polygons

    @property
    def mesh_area(self):
        return self._mesh_area


class _ClippedVoronoiGenerator:
    def __init__(self, point_density, seed):
        self._point_density = point_density
        self._seed = seed
        self._circle_r = 12.0
        self.circle_area = np.pi * self._circle_r**2

    @cached_property
    def _generating_shape(self):
        cx, cy = Coords.base_origin
        base_coords = np.linspace(cx - self._circle_r, cx + self._circle_r, 5)
        base_coords = [(x, cy) for x in base_coords]

        num_points = 10
        thetas = np.linspace(0, np.pi, num_points, endpoint=False)

        xs = cx + self._circle_r * np.cos(thetas)
        ys = cy + self._circle_r * np.sin(thetas)
        half_circle_coords = [point for point in zip(xs, ys)]
        coords = base_coords + half_circle_coords
        stacked_coords = np.vstack(coords)

        generating_shape = Polygon(stacked_coords)
        return generating_shape

    @cached_property
    def _points(self):
        bounds = self._generating_shape.bounds

        n_generating_points = int(self.circle_area * self._point_density)

        rng = np.random.default_rng(self._seed)
        xs = rng.uniform(bounds[0], bounds[2], n_generating_points)
        ys = rng.uniform(bounds[1], bounds[3], n_generating_points)
        points_array = np.vstack((xs, ys)).T
        points = shapely.points(points_array)
        inside_mask = shapely.contains(self._generating_shape, points)
        inside_points = points_array[inside_mask]
        return inside_points

    def _get_inside_mask(self, points_array):
        points = shapely.points(points_array)
        inside_mask = shapely.contains(self._generating_shape, points)
        return inside_mask

    @cached_property
    def _relaxed_points(self):
        n_iterations = 20
        for _ in range(n_iterations):
            vor = Voronoi(self._points)
            new_points = []

            for i, region_idx in enumerate(vor.point_region):
                vertex_inds = vor.regions[region_idx]
                # Infinite or empty region: use original point
                if (-1 in vertex_inds) or (len(vertex_inds) == 0):
                    new_points.append(self._points[i])
                else:
                    vertices = vor.vertices[vertex_inds]
                    centroid = vertices.mean(axis=0)

                    if self._get_inside_mask(centroid):
                        new_points.append(centroid)
                    # Point outside generating shape: use original point
                    else:
                        new_points.append(self._points[i])
            self._points = np.array(new_points)
        return self._points

    def _clip_polygons(self, all_poly_inds, all_vertices):
        clipped_polygons = []
        for poly_inds in all_poly_inds:
            polygon = all_vertices[poly_inds]
            poly = Polygon(polygon)
            poly = poly.intersection(self._generating_shape)

            poly_vertices = list(poly.exterior.coords)
            if poly_vertices:
                stacked_poly_vertices = np.vstack(poly_vertices)
                clipped_polygons.append(stacked_poly_vertices)

        return clipped_polygons

    @cached_property
    def polygons(self, radius=None):
        vor = Voronoi(self._relaxed_points)
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        all_polygon_inds = []
        new_vertices = vor.vertices.tolist()

        tissue_centroid = vor.points.mean(axis=0)
        if radius is None:
            radius = 2 * np.ptp(vor.points).max()

        all_ridges = defaultdict(list)
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges[p1].append((p2, v1, v2))
            all_ridges[p2].append((p1, v1, v2))

        for p1, region in enumerate(vor.point_region):
            poly_inds = vor.regions[region]

            # Finite region
            if all(idx >= 0 for idx in poly_inds):
                all_polygon_inds.append(poly_inds)

            # Infinite region
            else:
                ridges = all_ridges[p1]
                new_poly_inds = [idx for idx in poly_inds if idx >= 0]

                for p2, v1, v2 in ridges:
                    if v2 < 0:
                        v1, v2 = v2, v1
                    if v1 >= 0:
                        # Finite ridge: already in the region
                        continue

                    # Compute the missing endpoint of an infinite ridge
                    tangent = vor.points[p2] - vor.points[p1]
                    tangent /= np.linalg.norm(tangent)
                    normal = np.array([-tangent[1], tangent[0]])

                    midpoint = vor.points[[p1, p2]].mean(axis=0)
                    direction = (
                        np.sign(np.dot(midpoint - tissue_centroid, normal))
                        * normal
                    )
                    far_point = vor.vertices[v2] + direction * radius

                    new_poly_inds.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())

                unsorted_vertices = np.array(
                    [new_vertices[idx] for idx in new_poly_inds]
                )
                sorted_poly_inds = sort_counterclockwise(
                    new_poly_inds, unsorted_vertices
                )
                all_polygon_inds.append(sorted_poly_inds)

        all_vertices = np.array(new_vertices)

        clipped_polygons = self._clip_polygons(all_polygon_inds, all_vertices)

        return clipped_polygons


class _VoronoiPolygons(_Polygons):
    def __init__(self, params):
        self._clipped_voronoi_generator = _ClippedVoronoiGenerator(
            point_density=0.33, seed=params.seed
        )
        super().__init__()

    def _build(self):
        self._mesh_area = self._calc_mesh_area()
        self._all_polygon_inds, self._init_vertices = (
            self._make_init_polygons()
        )
        self._max_vertices = self._find_max_vertices()
        self._indices = self._finalize_polygon_inds()
        self._free_mask = self._get_free_mask()

    def _calc_mesh_area(self):
        area = 0.5 * self._clipped_voronoi_generator.circle_area
        return area

    def _make_shared_vertex_structure(self):
        vertex_to_index = {}
        unique_vertices: list[tuple] = []
        poly_indices = []

        for poly in self._clipped_voronoi_generator.polygons:
            # Remove duplicate last point if same as first
            if np.array_equal(poly[0], poly[-1]):
                poly = poly[:-1]

            poly_idx = []
            for pt in poly:
                pt_tuple = tuple(pt)
                if pt_tuple not in vertex_to_index:
                    vertex_to_index[pt_tuple] = len(unique_vertices)
                    unique_vertices.append(pt_tuple)
                poly_idx.append(vertex_to_index[pt_tuple])
            poly_indices.append(poly_idx)

        vertices = np.array(unique_vertices)
        return poly_indices, vertices

    def _sort_all_counterclockwise(self, all_poly_inds, all_vertices):
        all_sorted_poly_inds = []
        for poly_inds in all_poly_inds:
            vertices = all_vertices[poly_inds]
            sorted_inds = sort_counterclockwise(poly_inds, vertices)
            all_sorted_poly_inds.append(sorted_inds)
        return all_sorted_poly_inds

    def _extend_polygons(self, polygons):
        extended_polygons = []
        for polygon in polygons:
            extended_polygon = _extend(polygon)
            extended_polygons.append(extended_polygon)
        return extended_polygons

    def _make_init_polygons(self):
        all_polygon_inds, vertices = self._make_shared_vertex_structure()
        all_polygon_inds = self._sort_all_counterclockwise(
            all_polygon_inds, vertices
        )
        all_polygon_inds = self._extend_polygons(all_polygon_inds)
        return all_polygon_inds, vertices

    def _find_max_vertices(self):
        max_vertices = 0
        for vertex_inds in self._all_polygon_inds:
            max_vertices = max(max_vertices, len(vertex_inds))
        return max_vertices

    def _finalize_polygon_inds(self):
        all_polygon_inds = []
        for vertex_inds in self._all_polygon_inds:
            n_padding_values = self._max_vertices - len(vertex_inds)
            padding_array = np.full((n_padding_values,), -1, dtype=np.int64)
            polygon_inds = np.concatenate(
                [np.array(vertex_inds), padding_array]
            )
            all_polygon_inds.append(polygon_inds)

        all_polygon_inds = np.stack(all_polygon_inds)
        return all_polygon_inds

    def _get_free_mask(self):
        are_fixed = np.isclose(
            self._init_vertices[:, 1] - Coords.base_origin[1], 0.0
        )
        free_mask = np.ones(self._init_vertices.shape, dtype=bool)
        free_mask[are_fixed, 1] = False
        return free_mask


class _FullPolygons(_Polygons):
    def __init__(self):
        super().__init__()

    def _build(self):
        self._input_cells = self._read_input_cells()
        self._max_vertices = self._find_max_vertices()
        self._indices, self._init_vertices = self._make_init_polygons()
        self._free_mask = self._get_free_mask()
        self._mesh_area = self._calc_mesh_area()

    def _read_input_cells(self):
        input_path = files("diff_tissue.resources").joinpath(
            "input_cells.json"
        )
        with input_path.open() as data:
            input_cells = json.load(data)

        return input_cells

    def _find_max_vertices(self):
        max_vertices = 0
        for polygon in self._input_cells:
            if not polygon["is_boundary"]:
                max_vertices = max(max_vertices, len(polygon["edges"]))
        # Compensates for extra vertex added for efficiency
        max_vertices += 1
        return max_vertices

    @staticmethod
    def _remove_duplicates(lst):
        """Removes duplicates while preserving order."""
        unique = list(dict.fromkeys(lst))
        return unique

    def _make_init_polygons(self):
        all_vertices = np.zeros((0, 2))
        padded_list_of_indices = []
        index = 0
        for polygon in self._input_cells:
            if polygon["is_boundary"]:
                continue
            indices = []
            vertices = polygon["edges"]
            for vertex in vertices:
                are_equal = np.isclose(
                    np.array(vertex) - all_vertices, 0.0, atol=0.5
                )
                possible_inds = np.where(np.all(are_equal, axis=1))[0]

                # Add new index
                if len(possible_inds) == 0:
                    all_vertices = np.vstack([all_vertices, vertex])
                    indices.append(index)
                    index += 1
                # Use existing index
                elif len(possible_inds) == 1:
                    indices.append(int(possible_inds[0]))
                else:
                    raise ValueError("Multiple indices found")

            indices = self._remove_duplicates(indices)
            indices = sort_counterclockwise(indices, all_vertices[indices])
            # For efficiency
            indices = _extend(indices)

            # Pad
            indices += [-1] * (self._max_vertices - len(indices))
            indices.extend([-1] * (self._max_vertices - len(indices)))

            padded_list_of_indices.append(indices)

        all_indices = np.array(padded_list_of_indices)

        # Transform coordinates
        all_vertices -= Coords.full_mesh_base

        return all_indices, all_vertices

    def _get_fixed_inds(self):
        fixed_inds = [
            3,
            0,
            15,
            16,
            27,
            37,
            47,
            66,
            97,
            103,
            110,
            145,
            128,
            123,
            107,
            78,
            52,
            35,
            18,
            10,
            11,
            6,
            7,
        ]
        return fixed_inds

    def _get_free_mask(self):
        fixed_inds = self._get_fixed_inds()
        free_mask = np.ones(self._init_vertices.shape, dtype=bool)
        free_mask[fixed_inds] = False
        return free_mask

    @staticmethod
    def _calc_segment_area(d, r):
        segment_area = r**2 * np.arccos(d / r) - d * np.sqrt(r**2 - d**2)
        return segment_area

    def _calc_mesh_area(self):
        big_r = 40.0
        big_circle_area = np.pi * big_r**2
        center_to_base_line_dist = big_r - Coords.full_mesh_base[1]
        big_circle_segment = self._calc_segment_area(
            center_to_base_line_dist, big_r
        )

        small_r = 33.0
        center_to_base_line_dist = Coords.full_mesh_base[1]
        small_circle_segment = self._calc_segment_area(
            center_to_base_line_dist, small_r
        )
        mesh_area = big_circle_area - big_circle_segment - small_circle_segment
        return mesh_area


class _SinglePolygon(_Polygons):
    def __init__(self):
        super().__init__()

    def _build(self):
        self._init_vertices = self._make_init_polygons()
        self._n_vertices = self._init_vertices.shape[0]
        self._indices = self._find_polygon_inds()
        self._free_mask = self._get_free_mask()
        self._mesh_area = self._calc_mesh_area()

    def _make_init_polygons(self):
        vertices = Coords.base_origin + np.array(
            [
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 1.5],
                [1.0, 3.0],
                [-1.0, 3.0],
                [-2.0, 1.5],
            ]
        )
        return vertices

    def _find_polygon_inds(self):
        polygon_inds = np.arange(self._n_vertices)
        polygon_inds = sort_counterclockwise(polygon_inds, self._init_vertices)
        polygon_inds = _extend(polygon_inds)
        return np.array(polygon_inds).reshape(1, -1)

    def _get_free_mask(self):
        free_mask = np.ones(self._init_vertices.shape, dtype=bool)
        free_mask[:3, 1] = False
        return free_mask

    def _calc_mesh_area(self):
        xs = self._init_vertices[:, 0]
        ys = self._init_vertices[:, 1]

        area = 0.5 * np.abs(
            np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
        )
        return area


def get_system(params) -> _Polygons:
    polygons: _Polygons
    match params.system:
        case "voronoi":
            polygons = _VoronoiPolygons(params)
        case "full":
            polygons = _FullPolygons()
        case "single":
            polygons = _SinglePolygon()
        case _:
            raise ValueError("Invalid initial system!")
    return polygons


@struct.dataclass
class JaxPolygons:
    init_vertices: jnp.ndarray
    indices: jnp.ndarray
    free_mask: jnp.ndarray
    valid_mask: jnp.ndarray
    boundary_inds: jnp.ndarray
    poly_neighbors: jnp.ndarray
    vertex_neighbors: jnp.ndarray
    vertex_polygons: jnp.ndarray
    mesh_area: float


def get_jax_polygons(params):
    polygons = get_system(params)
    jax_polygons = JaxPolygons(
        init_vertices=jnp.array(polygons.init_vertices),
        indices=jnp.array(polygons.indices),
        free_mask=jnp.array(polygons.free_mask),
        valid_mask=jnp.array(polygons.valid_mask),
        boundary_inds=jnp.array(polygons.boundary_inds),
        poly_neighbors=jnp.array(polygons.poly_neighbors),
        vertex_neighbors=jnp.array(polygons.vertex_neighbors),
        vertex_polygons=jnp.array(polygons.vertex_polygons),
        mesh_area=polygons.mesh_area,
    )
    return jax_polygons


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


class VertexNumbers:
    def __init__(self, polygons):
        self._polygons = polygons

    @cached_property
    def basal(self):
        n_basal_vertices = np.isclose(
            self._polygons.free_mask[:, 1], 0.0
        ).sum()
        return n_basal_vertices

    @cached_property
    def boundary(self):
        n_boundary_vertices = len(self._polygons.boundary_inds)
        return n_boundary_vertices

    @cached_property
    def non_basal(self):
        n_non_basal_vertices = self.boundary - self.basal + 2
        return n_non_basal_vertices


class Knots:
    def __init__(self):
        self._nx_left = 1
        self._ny = 5
        self._xmin, self._xmax = -5.0, -5.0
        self._ymin, self._ymax = 1.0, 13.0

    def _make_grid(self) -> NDArray[np.floating]:
        x = np.linspace(self._xmin, self._xmax, self._nx_left)
        y = np.linspace(self._ymin, self._ymax, self._ny)
        grid = np.meshgrid(x, y, indexing="ij")
        return np.array(grid)

    @cached_property
    def left_knots(self):
        grid = self._make_grid()

        xs = grid[0].flatten()
        ys = grid[1].flatten()
        left_knots = np.column_stack([xs, ys])

        left_knots += Coords.base_origin
        return left_knots

    @cached_property
    def right_knots(self):
        flipped_array = np.empty_like(self.left_knots)
        shape_x = Coords.base_origin[0]
        flipped_array[:, 0] = 2 * shape_x - self.left_knots[:, 0]
        flipped_array[:, 1] = self.left_knots[:, 1]
        return flipped_array

    @cached_property
    def center_knots(self):
        center_knots = np.empty((self._ny, 2))
        center_knots[:, 0] = Coords.base_origin[0]
        center_knots[:, 1] = self.left_knots[: self._ny, 1]
        return center_knots

    @cached_property
    def all_knots(self):
        all_knots = np.concatenate(
            [self.left_knots, self.center_knots, self.right_knots], axis=0
        )
        return all_knots
