from abc import ABC, abstractmethod
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import Voronoi
import shapely
from shapely.geometry import Polygon


class Coords:
    _origin = np.array([0.0, 0.0])
    base_origin = _origin + 0.0
    shape_origin = _origin + np.array([0.0, 26.5])
    full_mesh_origin = np.array([40.0, 0.0])
    full_mesh_base = np.array([40.0, 18.635])


def sort_counterclockwise(indices, vertices):
    centroid = vertices.mean(axis=0)
    angles = np.arctan2(
        vertices[:,1] - centroid[1], vertices[:,0] - centroid[0]
    )
    inds_for_sorting = np.argsort(angles)
    sorted_poly_inds = np.array(indices)[inds_for_sorting]
    sorted_poly_inds = sorted_poly_inds.tolist()
    return sorted_poly_inds


def _extend(indices):
    indices.insert(0, indices[-1])
    indices.append(indices[1])
    return indices


def _calc_segment_area(d, r):
    segment_area = r**2 * np.arccos(d / r) - d * np.sqrt(r**2 - d**2)
    return segment_area


def _get_full_mesh_area():
    big_r = 40.0
    big_circle_area = np.pi * big_r**2
    center_to_base_line_dist = big_r - Coords.full_mesh_base[1]
    big_circle_segment = _calc_segment_area(center_to_base_line_dist, big_r)

    small_r = 33.0
    center_to_base_line_dist = Coords.full_mesh_base[1]
    small_circle_segment = _calc_segment_area(
        center_to_base_line_dist, small_r
    )
    full_mesh_area = big_circle_area - big_circle_segment - small_circle_segment
    print(f'Full mesh area = {full_mesh_area:.3f}')
    return full_mesh_area


def _calc_single_poly_area(vertices):
    xs = vertices[:,0]
    ys = vertices[:,1]

    area = 0.5 * np.abs(
        np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
    )
    print(f'Single polygon area = {area:.3f}')
    return area


class _Polygons(ABC):
    def __init__(self):
        self._build()

        self._valid_mask = self._find_valid_mask()
        self._boundary_mask = self._find_boundary_mask()
        self._centroids = self._calc_centroids()
        self._poly_neighbors = self._calc_poly_neighbors()
        self._vertex_neighbors = self._calc_vertex_neighbors()

    @abstractmethod
    def _build(self):
        pass

    def _find_valid_mask(self):
        valid_mask = (self._polygon_inds != -1)
        return valid_mask

    def _find_boundary_mask(self):
        all_edges = set()
        interior_edges = set()
        for extended_polygon in self._polygon_inds:
            unpadded_polygon = extended_polygon[extended_polygon != -1]
            # Removes the last vertex, which was an extension for efficiency
            polygon = unpadded_polygon[:-1]

            for i in range(len(polygon) - 1):
                edge = polygon[i:i+2]
                sorted_edge = tuple(np.sort(edge))
                if sorted_edge in all_edges:
                    interior_edges.add(sorted_edge)

                all_edges.add(sorted_edge)

        boundary_edges = all_edges - interior_edges
        boundary_inds = np.array(list(boundary_edges)).flatten()
        boundary_inds = np.unique(boundary_inds)
        boundary_mask = np.zeros(self._vertices.shape[0], dtype=bool)
        boundary_mask[boundary_inds] = True

        return boundary_mask

    def _calc_centroids(self):
        polygons = self._vertices[self._polygon_inds]
        mask = self._valid_mask[..., None].repeat(2, axis=2)
        polygons[~mask] = np.nan
        centroids = np.nanmean(polygons, axis=1)
        return centroids

    @staticmethod
    def _list_of_ints_to_padded_array(all_neighbors):
        max_n_neighbors = max([
            len(neighbors_list) for neighbors_list in all_neighbors
        ])
        padded_neighbors = -1 * np.ones(
            (len(all_neighbors), max_n_neighbors), dtype=int
        )
        for i, neighbors in enumerate(all_neighbors):
            n_poly_neighbors = len(neighbors)
            padded_neighbors[i,:n_poly_neighbors] = neighbors

        return padded_neighbors

    def _calc_poly_neighbors(self):
        neighbors = []
        max_n_neighbors = 0

        for idx, vertex_inds_ in enumerate(self._polygon_inds):
            vertex_inds = vertex_inds_[vertex_inds_ != -1]
            poly_neighbors = np.isin(self._polygon_inds, vertex_inds)
            poly_neighbors = np.where(np.any(poly_neighbors, axis=1))[0]
            poly_neighbors = poly_neighbors[poly_neighbors != idx]
            neighbors.append(poly_neighbors)

            n_neighbors = len(poly_neighbors)
            if n_neighbors > max_n_neighbors:
                max_n_neighbors = n_neighbors

        neighbors = self._list_of_ints_to_padded_array(neighbors)
        return neighbors

    def _calc_vertex_neighbors(self):
        vertex_neighbors = defaultdict(set)
        for polygon in self._polygon_inds:
            for i in range(len(polygon) - 1):
                vertex_idx, vertex_neighbor_idx = (
                    int(polygon[i]), int(polygon[i+1])
                )

                if not vertex_neighbor_idx == -1:
                    vertex_neighbors[vertex_idx].add(int(vertex_neighbor_idx))
                    vertex_neighbors[vertex_neighbor_idx].add(int(vertex_idx))

        vertex_neighbors = [
            list(neighbors_set) for neighbors_set in vertex_neighbors.values()
        ]
        vertex_neighbors = self._list_of_ints_to_padded_array(vertex_neighbors)
        return vertex_neighbors

    @property
    def vertices(self):
        return self._vertices

    @property
    def polygon_inds(self):
        return self._polygon_inds

    @property
    def free_mask(self):
        return self._free_mask

    @property
    def proximal_mask(self):
        return self._proximal_mask

    @property
    def valid_mask(self):
        return self._valid_mask

    @property
    def boundary_mask(self):
        return self._boundary_mask

    @property
    def centroids(self):
        return self._centroids

    @property
    def poly_neighbors(self):
        return self._poly_neighbors

    @property
    def vertex_neighbors(self):
        return self._vertex_neighbors


class _MeshPolygons(_Polygons):
    def __init__(self):
        super().__init__()

    def _build(self):
        self._input_cells = self._read_input_cells()
        self._max_vertices = self._find_max_vertices()
        self._polygon_inds, self._vertices, self._proximal_mask = (
            self._make_init_polygons()
        )
        self._free_mask = self._get_free_mask()

    def _read_input_cells(self):
        input_path = Path('input_cells.json')
        with input_path.open() as data:
            input_cells = json.load(data)

        return input_cells

    def _find_max_vertices(self):
        max_vertices = 0
        for polygon in self._input_cells:
            if not polygon['is_boundary']:
                max_vertices = max(max_vertices, len(polygon['edges']))
        # Compensates for extra vertex added for efficiency
        max_vertices += 1
        return max_vertices

    def _is_proximal(self, vertex):
        dist_from_origin = np.linalg.norm(vertex - Coords.full_mesh_origin)
        proximal_radius = np.inf
        return dist_from_origin < proximal_radius

    @staticmethod
    def _remove_duplicates(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    def _make_init_polygons(self):
        all_vertices = np.zeros((0, 2))
        all_indices = []
        proximal_mask = []
        index = 0
        for polygon in self._input_cells:
            if polygon['is_boundary']:
                continue
            indices = []
            vertices = polygon['edges']
            is_proximal = True
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
                    raise ValueError('Multiple indices found')

                # True if all vertices are proximal
                is_proximal *= self._is_proximal(vertex)

            indices = self._remove_duplicates(indices)
            indices = sort_counterclockwise(
                indices, all_vertices[indices]
            )
            # For efficiency
            indices = _extend(indices)

            # Pad
            indices += [-1] * (self._max_vertices - len(indices))
            indices.extend([-1] * (self._max_vertices - len(indices)))

            all_indices.append(indices)
            proximal_mask.append(is_proximal)

        all_indices = np.array(all_indices)
        proximal_mask = np.array(proximal_mask)

        # Transform coordinates
        all_vertices -= Coords.full_mesh_base

        return all_indices, all_vertices, proximal_mask

    def _get_fixed_inds(self):
        fixed_inds = [
            3, 0, 15, 16, 27, 37, 47, 66, 97, 103, 110, 145, 128, 123, 107, 78,
            52, 35, 18, 10, 11, 6, 7
        ]
        return fixed_inds

    def _get_free_mask(self):
        fixed_inds = self._get_fixed_inds()
        free_mask = np.ones(self._vertices.shape, dtype=bool)
        free_mask[fixed_inds] = False
        return free_mask


class _VoronoiPolygons(_Polygons):
    def __init__(self):
        super().__init__()

    def _build(self):
        self._radius_x = 12.0
        self._radius_y = 12.0
        self._polygon_area = 3.0
        self._n_polygons_in_full_circle = self._calc_n_polygons_in_full_circle()
        self._mesh_area = self._calc_mesh_area()

        self._generating_shape = self._get_generating_shape()
        self._all_polygon_inds, self._vertices = (
            self._make_init_polygons()
        )
        self._max_vertices = self._find_max_vertices()
        self._polygon_inds = self._finalize_polygon_inds()
        self._free_mask = self._get_free_mask()
        self._proximal_mask = np.ones(self._polygon_inds.shape[0], dtype=bool)

    def _get_generating_shape(self):
        cx, cy = Coords.base_origin
        base_coords = np.linspace(cx - self._radius_x, cx + self._radius_x, 5)
        base_coords = [(x, cy) for x in base_coords]

        num_points = 10
        thetas = np.linspace(0, np.pi, num_points, endpoint=False)

        xs = cx + self._radius_x * np.cos(thetas)
        ys = cy + self._radius_y * np.sin(thetas)
        half_circle_coords = [point for point in zip(xs, ys)]
        coords = base_coords + half_circle_coords
        coords = np.vstack(coords)

        generating_shape = Polygon(coords)
        return generating_shape

    def _calc_n_polygons_in_full_circle(self):
        n_polygons = int(
            np.pi * self._radius_x * self._radius_y /
            self._polygon_area
        )
        return n_polygons

    def _calc_mesh_area(self):
        area = 0.5 * self._n_polygons_in_full_circle * self._polygon_area
        print(f'Voronoi mesh area = {area:.3f}')
        return area

    def _generate_random_points(self):
        bounds = self._generating_shape.bounds
        xs = np.random.uniform(
            bounds[0], bounds[2], self._n_polygons_in_full_circle
        )
        ys = np.random.uniform(
            bounds[1], bounds[3], self._n_polygons_in_full_circle
        )
        points_array = np.vstack((xs, ys)).T
        points = shapely.points(points_array)
        inside_mask = shapely.contains(self._generating_shape, points)
        inside_points = points_array[inside_mask]

        return inside_points

    def _get_inside_mask(self, points_array):
        points = shapely.points(points_array)
        inside_mask = shapely.contains(self._generating_shape, points)
        return inside_mask

    def _get_inside_points(self, points_array):
        inside_mask = self._get_inside_mask(points_array)
        inside_points = points_array[inside_mask]
        return inside_points

    def _generate_relaxed_random_points(self):
        points = self._generate_random_points()
        n_iterations = 20
        for _ in range(n_iterations):
            vor = Voronoi(points)
            new_points = []

            for i, region_idx in enumerate(vor.point_region):
                vertex_inds = vor.regions[region_idx]
                # Infinite or empty region: use original point
                if (-1 in vertex_inds) or (len(vertex_inds) == 0):
                    new_points.append(points[i])
                else:
                    vertices = vor.vertices[vertex_inds]
                    centroid = vertices.mean(axis=0)

                    if self._get_inside_mask(centroid):
                        new_points.append(centroid)
                    # Point outside generating shape: use original point
                    else:
                        new_points.append(points[i])
            points = np.array(new_points)

        return points

    def _clip_polygons(self, all_poly_inds, all_vertices):
        clipped_polygons = []
        for poly_inds in all_poly_inds:
            polygon = all_vertices[poly_inds]
            poly = Polygon(polygon)
            poly = poly.intersection(self._generating_shape)

            poly_vertices = list(poly.exterior.coords)
            if poly_vertices:
                poly_vertices = np.vstack(poly_vertices)
                clipped_polygons.append(poly_vertices)

        return clipped_polygons

    def _make_clipped_voronoi(self, relaxed_points, radius=None):
        vor = Voronoi(relaxed_points)
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
                    direction = np.sign(
                        np.dot(midpoint - tissue_centroid, normal)
                    ) * normal
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

    @staticmethod
    def _make_shared_vertex_structure(polygons):
        vertex_to_index = {}
        unique_vertices = []
        poly_indices = []

        for poly in polygons:
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

    def _separate_close_vertices(self, vertices):
        min_dist = 0.5 * np.sqrt(self._polygon_area)
        jitter_scale = 0.1 * min_dist
        max_n = 1000
        for n in range(max_n):
            diffs = vertices[:,None,:] - vertices
            dists = np.linalg.norm(diffs, axis=-1)
            np.fill_diagonal(dists, np.inf)

            too_close = (dists < min_dist) & (dists > 0.0)
            n_too_close = too_close.sum()
            if n_too_close == 0:
                break
            if n == max_n - 1:
                print(
                    f'Jittering did not converge after {max_n} iterations.'
                )
                sys.exit()

            for i in range(dists.shape[0]):
                for j in range(i+1, dists.shape[1]):
                    if dists[i,j] < min_dist:
                        norm_diff_vec = diffs[i,j] / np.linalg.norm(diffs[i,j])
                        jitter = 0.5 * jitter_scale * norm_diff_vec
                        vertices[i] += jitter
                        vertices[j] -= jitter

        return vertices

    def _sort_all_counterclockwise(self, all_poly_inds, all_vertices):
        all_sorted_poly_inds = []
        for poly_inds in all_poly_inds:
            vertices = all_vertices[poly_inds]
            sorted_inds = sort_counterclockwise(
                poly_inds, vertices
            )
            all_sorted_poly_inds.append(sorted_inds)
        return all_sorted_poly_inds

    def _extend_polygons(self, polygons):
        extended_polygons = []
        for polygon in polygons:
            extended_polygon = _extend(polygon)
            extended_polygons.append(extended_polygon)
        return extended_polygons

    def _make_init_polygons(self):
        relaxed_points = self._generate_relaxed_random_points()
        polygons = self._make_clipped_voronoi(relaxed_points)
        all_polygon_inds, vertices = self._make_shared_vertex_structure(
            polygons
        )
        all_polygon_inds = self._sort_all_counterclockwise(
            all_polygon_inds, vertices
        )
        all_polygon_inds = self._extend_polygons(all_polygon_inds)

        # Commented out to keep base vertices exactly at the base line.
        # If the program remains numerically stable,
        # this can eventually be deleted entirely.

        # vertices = self._separate_close_vertices(vertices)

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
        are_fixed = np.isclose(self._vertices[:,1] - Coords.base_origin[1], 0.0)
        free_mask = np.ones(self._vertices.shape, dtype=bool)
        free_mask[are_fixed, 1] = False
        return free_mask


class _SinglePolygon(_Polygons):
    def __init__(self):
        super().__init__()

    def _build(self):
        self._vertices = self._make_init_polygons()
        self._n_vertices = self._vertices.shape[0]
        self._polygon_inds = self._find_polygon_inds()
        self._free_mask = self._get_free_mask()
        self._proximal_mask = np.array([True])

    def _make_init_polygons(self):
        vertices = Coords.base_origin + np.array([
            [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 1.5], [1.0, 3.0],
            [-1.0, 3.0], [-2.0, 1.5]
        ])
        return vertices

    def _find_polygon_inds(self):
        polygon_inds = np.arange(self._n_vertices)
        polygon_inds = sort_counterclockwise(
            polygon_inds, self._vertices
        )
        polygon_inds = _extend(polygon_inds)
        return np.array(polygon_inds).reshape(1, -1)

    def _get_free_mask(self):
        free_mask = np.ones(self._vertices.shape, dtype=bool)
        free_mask[:3,1] = False
        return free_mask


class _AbstractFactory(ABC):
    def __init__(self, system):
        self._build()
        self._system = system
        self._shape_params, self._polygons = self._make_params_and_polygons()
        self._vertex_numbers = self._find_vertex_numbers()
        self._outer_shape = self._make_outer_shape()

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def _make_params_and_polygons(self):
        pass

    def _find_vertex_numbers(self):
        n_basal_vertices = (
            np.isclose(self._polygons.free_mask[:,1], 0.0).sum()
        )
        n_boundary_vertices = self._polygons.boundary_mask.sum()
        n_non_basal_vertices = n_boundary_vertices - n_basal_vertices + 2
        vertex_numbers = {
            'basal': n_basal_vertices,
            'boundary': n_boundary_vertices,
            'non_basal': n_non_basal_vertices
        }
        return vertex_numbers

    @staticmethod
    def _resample_curve(points, n_points, spacing=None):
        points = np.asarray(points)
        diffs = np.diff(points, axis=0)
        seg_lengths = np.hypot(diffs[:,0], diffs[:,1])
        cumulative_lengths = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        total_length = cumulative_lengths[-1]

        if spacing is not None:
            n_points = int(np.floor(total_length / spacing)) + 1
        elif n_points is None:
            raise ValueError('You must provide either num_points or spacing.')

        target_lengths = np.linspace(0, total_length, n_points)

        resampled = []
        for t in target_lengths:
            idx = np.searchsorted(cumulative_lengths, t) - 1
            idx = min(max(idx, 0), len(points) - 2)
            t0, t1 = cumulative_lengths[idx], cumulative_lengths[idx+1]
            p0, p1 = points[idx], points[idx+1]
            # Handle zero-length segments
            if t1 == t0:
                resampled.append(p0)
            else:
                alpha = (t - t0) / (t1 - t0)
                resampled.append((1 - alpha)*p0 + alpha*p1)
        return np.array(resampled)

    def _make_non_basal_vertices(self, xs, ys):
        vertices = np.array(
            [(x, y) for x, y in zip(xs, ys)]
        )
        vertices = self._resample_curve(
            vertices, self._vertex_numbers['non_basal']
        )
        return vertices

    def _make_basal_vertices(self, lower_r):
        basal_xs = np.linspace(
            -lower_r, lower_r, self._vertex_numbers['basal']
        )[1:-1]
        basal_vertices = np.array([(x, 0.0) for x in basal_xs])
        return basal_vertices

    def _finalize_vertices(self, non_basal_vertices, basal_vertices):
        vertices = np.concatenate([non_basal_vertices, basal_vertices], axis=0)
        vertices += Coords.base_origin
        return vertices

    def _construct_outer_shape(self, non_basal_xs, non_basal_ys, lower_r):
        non_basal_vertices = self._make_non_basal_vertices(
            non_basal_xs, non_basal_ys
        )
        basal_vertices = self._make_basal_vertices(lower_r)
        outer_shape = self._finalize_vertices(
            non_basal_vertices, basal_vertices
        )
        return outer_shape

    @abstractmethod
    def _make_outer_shape(self):
        pass

    @property
    def polygons(self):
        return self._polygons

    @property
    def outer_shape(self):
        return self._outer_shape


class _TrapzeoidFactory(_AbstractFactory):
    def __init__(self, system):
        super().__init__(system)

    def _build(self):
        self._height = 3.5
        self._lower_r = 1.5
        self._upper_r = 2.0

    def _calc_scale(self, mesh_area):
        scale = np.sqrt(mesh_area / self._height**2)
        return scale

    def _make_params_and_polygons(self):
        match self._system:
            case 'full':
                mesh_area = _get_full_mesh_area()
                scale = self._calc_scale(mesh_area)
                polygons = _MeshPolygons()
            case 'voronoi':
                mesh_area = 225.0 # Manually insert for now
                scale = self._calc_scale(mesh_area)
                polygons = _VoronoiPolygons()
            case 'single':
                raise NotImplementedError
            case _:
                raise ValueError('Invalid initial system!')

        shape_params = {'scale': scale}
        return shape_params, polygons

    def _make_outer_shape(self):
        scaled_height = self._height * self._shape_params['scale']
        scaled_lower_r = self._lower_r * self._shape_params['scale']
        scaled_upper_r = self._upper_r * self._shape_params['scale']

        non_basal_xs = np.array(
            [scaled_lower_r, scaled_upper_r, -scaled_upper_r, -scaled_lower_r]
        )
        non_basal_ys = np.array([0.0, scaled_height, scaled_height, 0.0])

        outer_shape = self._construct_outer_shape(
            non_basal_xs, non_basal_ys, scaled_lower_r
        )

        area = scaled_height * (scaled_lower_r + scaled_upper_r)
        print(f'Outer shape area = {area:.3f}')

        return outer_shape


class _TriangleFactory(_AbstractFactory):
    def __init__(self, system):
        super().__init__(system)

    def _build(self):
        self._height = 2.5
        self._lower_r = 1.5

    def _calc_scale(self, mesh_area):
        scale = np.sqrt(mesh_area / (self._height * self._lower_r))
        return scale

    def _make_params_and_polygons(self):
        match self._system:
            case 'full':
                mesh_area = _get_full_mesh_area()
                scale = self._calc_scale(mesh_area)
                polygons = _MeshPolygons()
            case 'voronoi':
                mesh_area = 225.0 # Manually insert for now
                scale = self._calc_scale(mesh_area)
                polygons = _VoronoiPolygons()
            case 'single':
                raise NotImplementedError
            case _:
                raise ValueError('Invalid initial system!')

        shape_params = {'scale': scale}
        return shape_params, polygons

    def _make_outer_shape(self):
        scaled_height = self._height * self._shape_params['scale']
        scaled_lower_r = self._lower_r * self._shape_params['scale']

        non_basal_xs = np.array([scaled_lower_r, 0.0, -scaled_lower_r])
        non_basal_ys = np.array([0.0, scaled_height, 0.0])

        outer_shape = self._construct_outer_shape(
            non_basal_xs, non_basal_ys, scaled_lower_r
        )

        area = scaled_height * scaled_lower_r
        print(f'Outer shape area = {area:.3f}')

        return outer_shape


class _PetalFactory(_AbstractFactory):
    def __init__(self, system):
        super().__init__(system)

    def _build(self):
        self._lower_r = 20.0
        self._height = 60.0
        self._stretch_strength = 2.0

    def _calc_scale(self, mesh_area):
        scale = np.sqrt(
            mesh_area / ((self._lower_r * self._height) *
            (np.pi / 2 + 2 * self._stretch_strength / 3))
        )
        return scale

    def _make_params_and_polygons(self):
        match self._system:
            case 'full':
                mesh_area = _get_full_mesh_area()
                scale = self._calc_scale(mesh_area)
                polygons = _MeshPolygons()
            case 'voronoi':
                mesh_area = 225.0 # Manually insert for now
                scale = self._calc_scale(mesh_area)
                polygons = _VoronoiPolygons()
            case 'single':
                polygons = _SinglePolygon()
                vertices = polygons.vertices
                mesh_area = _calc_single_poly_area(vertices)
                scale = self._calc_scale(mesh_area)
            case _:
                raise ValueError('Invalid initial system!')

        shape_params = {'scale': scale}
        return shape_params, polygons

    def _make_outer_shape(self):
        scaled_lower_r = self._lower_r * self._shape_params['scale']
        scaled_height = self._height * self._shape_params['scale']

        xs = np.linspace(
            -scaled_lower_r, scaled_lower_r, self._vertex_numbers['non_basal']
        )
        non_basal_ys = scaled_height * np.sqrt(1 - (xs / scaled_lower_r)**2)

        factor = 1 + self._stretch_strength * (non_basal_ys / scaled_height)
        non_basal_xs = xs * factor

        outer_shape = self._construct_outer_shape(
            non_basal_xs, non_basal_ys, scaled_lower_r
        )

        area = (
            scaled_lower_r * scaled_height *
            (np.pi / 2 + 2 * self._stretch_strength / 3)
        )
        print(f'Outer shape area = {area:.3f}')

        return outer_shape


def get_factory(shape, system):
    match shape:
        case 'trapezoid':
            factory = _TrapzeoidFactory(system)
        case 'triangle':
            factory = _TriangleFactory(system)
        case 'petal':
            factory = _PetalFactory(system)
        case _:
            raise ValueError('Invalid outer shape!')
    return factory
