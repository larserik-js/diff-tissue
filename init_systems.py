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


def _sort_counterclockwise(indices, vertices):
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


class _Polygons:
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

    def _calc_poly_neighbors(self):
        neighbors_ = []
        max_n_neighbors = 0
        n_polygons = self._polygon_inds.shape[0]

        for idx, vertex_inds_ in enumerate(self._polygon_inds):
            vertex_inds = vertex_inds_[vertex_inds_ != -1]
            poly_neighbors = np.isin(self._polygon_inds, vertex_inds)
            poly_neighbors = np.where(np.any(poly_neighbors, axis=1))[0]
            poly_neighbors = poly_neighbors[poly_neighbors != idx]
            neighbors_.append(poly_neighbors)

            n_neighbors = len(poly_neighbors)
            if n_neighbors > max_n_neighbors:
                max_n_neighbors = n_neighbors

        neighbors = -1 * np.ones((n_polygons, max_n_neighbors), dtype=int)
        for i, poly_neighbors in enumerate(neighbors_):
            n_poly_neighbors = len(poly_neighbors)
            neighbors[i,:n_poly_neighbors] = poly_neighbors

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

        return vertex_neighbors

    @property
    def polygon_inds(self):
        return self._polygon_inds

    @property
    def valid_mask(self):
        return self._valid_mask

    @property
    def vertices(self):
        return self._vertices

    @property
    def centroids(self):
        return self._centroids

    @property
    def poly_neighbors(self):
        return self._poly_neighbors

    @property
    def free_mask(self):
        return self._free_mask

    @property
    def basal_mask(self):
        return self._basal_mask

    @property
    def boundary_mask(self):
        return self._boundary_mask


class _MeshPolygons(_Polygons):
    def __init__(self):
        self._input_cells = self._read_input_cells()
        self._max_vertices = self._find_max_vertices()
        self._polygon_inds, self._vertices, self._basal_mask = (
            self._make_init_polygons()
        )
        self._valid_mask = self._find_valid_mask()
        self._free_mask = self._get_free_mask()
        self._boundary_mask = self._find_boundary_mask()
        self._centroids = self._calc_centroids()
        self._poly_neighbors = self._calc_poly_neighbors()
        self._vertex_neighbors = self._calc_vertex_neighbors()

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

    def _is_basal(self, vertex):
        dist_from_origin = np.linalg.norm(vertex - Coords.full_mesh_origin)
        basal_radius = np.inf
        return dist_from_origin < basal_radius

    @staticmethod
    def _remove_duplicates(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    def _make_init_polygons(self):
        all_vertices = np.zeros((0, 2))
        all_indices = []
        basal_mask = []
        index = 0
        for polygon in self._input_cells:
            if polygon['is_boundary']:
                continue
            indices = []
            vertices = polygon['edges']
            is_basal = True
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

                # True if all vertices are basal
                is_basal *= self._is_basal(vertex)

            indices = self._remove_duplicates(indices)
            indices = _sort_counterclockwise(
                indices, all_vertices[indices]
            )
            # For efficiency
            indices = _extend(indices)

            # Pad
            indices += [-1] * (self._max_vertices - len(indices))
            indices.extend([-1] * (self._max_vertices - len(indices)))

            all_indices.append(indices)
            basal_mask.append(is_basal)

        all_indices = np.array(all_indices)
        basal_mask = np.array(basal_mask)

        # Transform coordinates
        all_vertices -= Coords.full_mesh_base

        return all_indices, all_vertices, basal_mask

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
        self._radius_x = 12.0
        self._radius_y = 12.0
        self._polygon_area = 3.0
        self._n_polygons = self._calc_n_polygons()
        print(f'Tissue area = {self._n_polygons * self._polygon_area}')

        self._generating_shape = self._get_generating_shape()
        self._all_polygon_inds, self._vertices = (
            self._make_init_polygons()
        )
        self._max_vertices = self._find_max_vertices()
        self._polygon_inds = self._finalize_polygon_inds()
        self._valid_mask = self._find_valid_mask()
        self._free_mask = self._get_free_mask()
        self._basal_mask = np.ones(self._polygon_inds.shape[0], dtype=bool)
        self._boundary_mask = self._find_boundary_mask()
        self._centroids = self._calc_centroids()
        self._poly_neighbors = self._calc_poly_neighbors()
        self._vertex_neighbors = self._calc_vertex_neighbors()

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

    def _calc_n_polygons(self):
        n_polygons = int(
            np.pi * self._radius_x * self._radius_y /
            self._polygon_area
        )
        return n_polygons

    def _generate_random_points(self):
        bounds = self._generating_shape.bounds
        xs = np.random.uniform(bounds[0], bounds[2], self._n_polygons)
        ys = np.random.uniform(bounds[1], bounds[3], self._n_polygons)
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
                sorted_poly_inds = _sort_counterclockwise(
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
            sorted_inds = _sort_counterclockwise(
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
        free_mask = np.ones_like(self._vertices)
        free_mask = np.ones(self._vertices.shape, dtype=bool)
        free_mask[are_fixed, 1] = False
        return free_mask


class _SinglePolygon(_Polygons):
    def __init__(self):
        self._vertices = self._make_init_polygons()
        self._n_vertices = self._vertices.shape[0]
        self._polygon_inds = self._find_polygon_inds()
        self._valid_mask = self._find_valid_mask()
        self._free_mask = self._get_free_mask()
        self._basal_mask = np.array([True])
        self._boundary_mask = self._find_boundary_mask()
        self._centroids = self._calc_centroids()
        self._poly_neighbors = self._calc_poly_neighbors()
        self._vertex_neighbors = self._calc_vertex_neighbors()

    def _make_init_polygons(self):
        vertices = Coords.base_origin + np.array([
            [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 1.5], [1.0, 3.0],
            [-1.0, 3.0], [-2.0, 1.5]
        ])
        return vertices

    def _find_polygon_inds(self):
        polygon_inds = np.arange(self._n_vertices)
        polygon_inds = _sort_counterclockwise(
            polygon_inds, self._vertices
        )
        polygon_inds = _extend(polygon_inds)
        return np.array(polygon_inds).reshape(1, -1)

    def _get_free_mask(self):
        free_mask = np.ones(self._vertices.shape, dtype=bool)
        free_mask[:3,1] = False
        return free_mask


class _AbstractFactory(ABC):
    @abstractmethod
    def _make_params_and_polygons(self):
        pass

    @property
    def polygons(self):
        return self._polygons

    @property
    def outer_shape(self):
        return self._outer_shape


class _EllipseFactory(_AbstractFactory):
    def __init__(self, system):
        self._system = system
        self._shape_params, self._polygons = self._make_params_and_polygons()
        self._outer_shape = self._make_outer_shape()

    def _make_params_and_polygons(self):
        match self._system:
            case 'full':
                a = 40.0
                b = a * 1.5
                origin = Coords.shape_origin + (0.0, 25.0)
                polygons = _MeshPolygons()
            case 'voronoi':
                a = 23.0
                b = a * 1.2
                origin = Coords.shape_origin
                polygons = _VoronoiPolygons()
            case 'single':
                a = 15.0
                b = a * 1.5
                origin = Coords.shape_origin
                polygons = _SinglePolygon()
            case _:
                raise ValueError('Invalid initial system!')

        shape_params = {'a': a, 'b': b, 'origin': origin}
        return shape_params, polygons

    def _make_outer_shape(self):
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=True)
        xs = (self._shape_params['origin'][0] +
             self._shape_params['a'] * np.cos(angles))
        ys = (self._shape_params['origin'][1] +
             self._shape_params['b'] * np.sin(angles))
        ellipse = np.stack([xs, ys], axis=1)
        return ellipse


class _TrapzeoidFactory(_AbstractFactory):
    def __init__(self, system):
        self._system = system
        self._shape_params, self._polygons = self._make_params_and_polygons()
        self._outer_shape = self._make_outer_shape()

    def _make_params_and_polygons(self):
        match self._system:
            case 'full':
                a = 2500.0
                origin = Coords.shape_origin + (0.0, 15.0)
                polygons = _MeshPolygons()
            case 'voronoi':
                a = 12.0
                origin = Coords.shape_origin + (0.0, -22.0)
                polygons = _VoronoiPolygons()
            case 'single':
                a = 10.0
                origin = Coords.shape_origin
                polygons = _SinglePolygon()
            case _:
                raise ValueError('Invalid initial system!')

        shape_params = {'scale': a, 'origin': origin}
        return shape_params, polygons

    def _make_outer_shape(self):
        height = 3.5
        xs = (
            np.array([-1.5, 1.5, 2.0, -2.0, -1.5]) * self._shape_params['scale']
            + self._shape_params['origin'][0]
        )
        ys = (
            np.array([0.0, 0.0, height, height, 0.0]) *
            self._shape_params['scale']
            + self._shape_params['origin'][1]
        )
        triangle = np.stack([xs, ys], axis=1)
        return triangle


class _PetalFactory(_AbstractFactory):
    def __init__(self, system):
        self._system = system
        self._shape_params, self._polygons = self._make_params_and_polygons()
        self._outer_shape = self._make_outer_shape()

    def _make_params_and_polygons(self):
        match self._system:
            case 'full':
                a = 2500.0
                origin = Coords.shape_origin + (0.0, 15.0)
                polygons = _MeshPolygons()
            case 'voronoi':
                a = 700.0
                origin = Coords.shape_origin + (0.0, 1.0)
                polygons = _VoronoiPolygons()
            case 'single':
                a = 700.0
                origin = Coords.shape_origin + (0.0, 1.0)
                polygons = _SinglePolygon()
            case _:
                raise ValueError('Invalid initial system!')

        b = 1.0 * a
        m = 3.0
        n1 = 30.0
        n2 = 15.0
        n3 = 15.0
        shape_params = {
            'a': a, 'b': b, 'm': m, 'n1': n1, 'n2': n2, 'n3': n3,
            'origin': origin
        }
        return shape_params, polygons

    @staticmethod
    def _resample_curve(points, num_points=None, spacing=None):
        points = np.asarray(points)
        diffs = np.diff(points, axis=0)
        seg_lengths = np.hypot(diffs[:,0], diffs[:,1])
        cumulative_lengths = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        total_length = cumulative_lengths[-1]

        if spacing is not None:
            num_points = int(np.floor(total_length / spacing)) + 1
        elif num_points is None:
            raise ValueError('You must provide either num_points or spacing.')

        target_lengths = np.linspace(0, total_length, num_points)

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

    def _make_outer_shape(self):
        scale = 0.254
        rx = 20.0 * scale
        h = 60.0 * scale

        n_base_vertices = (
            np.isclose(self._polygons.free_mask[:,1], 0.0).sum()
        )
        n_boundary_vertices = self._polygons.boundary_mask.sum()
        n_non_base_vertices = n_boundary_vertices - n_base_vertices + 2

        xs = np.linspace(-rx, rx, n_non_base_vertices)
        ys = h * np.sqrt(1 - (xs / rx)**2)

        stretch_strength = 2.0

        factor = 1 + stretch_strength * (ys / h)
        xs = xs * factor

        petal_vertices = np.array([(x, y) for x, y in zip(xs, ys)])
        petal_vertices = self._resample_curve(
            petal_vertices, num_points=len(petal_vertices)
        )

        base_xs = np.linspace(-rx, rx, n_base_vertices)[1:-1]
        base_vertices = np.array([(x, 0.0) for x in base_xs])

        vertices = np.concatenate([petal_vertices, base_vertices], axis=0)
        vertices += Coords.base_origin

        area = 2 * rx * h * (np.pi / 2 + 2 * stretch_strength / 3)

        print(f'Outer shape area = {area}')

        return vertices


def get_factory(shape, system):
    match shape:
        case 'ellipse':
            factory = _EllipseFactory(system)
        case 'trapezoid':
            factory = _TrapzeoidFactory(system)
        case 'petal':
            factory = _PetalFactory(system)
        case _:
            raise ValueError('Invalid outer shape!')
    return factory
