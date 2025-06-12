from abc import ABC, abstractmethod
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon


class _Polygons:
    def _check_convex(self, indices, all_vertices):
        polygon = all_vertices[indices]
        edges = polygon[1:] - polygon[:-1]
        cross_products = np.cross(edges[:-1], edges[1:])
        non_zero_cross_products = cross_products[
            ~np.isclose(cross_products, 0.0)
        ]
        signs = np.sign(non_zero_cross_products)
        if np.all(signs > 0.0):
            convex = True
        elif np.all(signs < 0.0):
            convex = False
        else:
            raise ValueError('Non-consistent ordering of vertices.')
        return convex

    def _sort_to_counterclockwise(self, indices, all_vertices):
        is_convex = self._check_convex(indices, all_vertices)
        if not is_convex:
            indices = indices[::-1]
        return indices

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

    def get_polygon_inds(self):
        return self._polygon_inds

    def get_valid_mask(self):
        return self._valid_mask

    def get_vertices(self):
        return self._vertices

    def get_centroids(self):
        centroids = self._calc_centroids()
        return centroids

    def get_fixed_mask(self):
        return self._fixed_mask

    def get_basal_mask(self):
        return self._basal_mask

    def get_boundary_mask(self):
        return self._boundary_mask


class _MeshPolygons(_Polygons):
    def __init__(self):
        self._input_cells = self._read_input_cells()
        self._max_vertices = self._find_max_vertices()
        self._polygon_inds, self._vertices, self._basal_mask = (
            self._make_init_polygons()
        )
        self._valid_mask = (self._polygon_inds != -1)
        self._fixed_mask = self._get_fixed_mask()
        self._boundary_mask = self._find_boundary_mask()

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
        pseudo_origin = np.array([40.0, 0.0])
        dist_from_pseudo_origin = np.linalg.norm(vertex - pseudo_origin)
        # Basal radius of 45.0 is good for Hibiscus trionum
        # (using full mesh)
        basal_radius = np.inf
        return dist_from_pseudo_origin < basal_radius

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
                    indices.append(possible_inds[0])
                else:
                    raise ValueError('Multiple indices found')

                # True if all vertices are basal
                is_basal *= self._is_basal(vertex)

            # For efficiency
            first_idx = indices[1]
            indices.append(first_idx)

            indices = self._sort_to_counterclockwise(indices, all_vertices)

            # Pad
            indices += [-1] * (self._max_vertices - len(indices))
            indices.extend([-1] * (self._max_vertices - len(indices)))

            all_indices.append(indices)
            basal_mask.append(is_basal)

        all_indices = np.array(all_indices)
        basal_mask = np.array(basal_mask)

        return all_indices, all_vertices, basal_mask

    def _get_fixed_inds(self):
        fixed_inds = [
            3, 0, 15, 16, 27, 37, 47, 66, 97, 103, 110, 145, 128, 123, 107, 78,
            52, 35, 18, 10, 11, 6, 7
        ]
        return fixed_inds

    def _get_fixed_mask(self):
        fixed_inds = self._get_fixed_inds()
        fixed_mask = np.ones_like(self._vertices)
        fixed_mask[fixed_inds] = 0.0
        return fixed_mask


class _SimpleMeshPolygons(_MeshPolygons):
    def __init__(self):
        self._selection_origin = np.array([40.0, 40.0])
        self._selection_radius = 20.0
        super().__init__()

    def _all_within_selection(self, vertices):
        dists = np.linalg.norm(vertices - self._selection_origin, axis=1)
        return np.all(dists < self._selection_radius)

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
            if not self._all_within_selection(vertices):
                continue
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
                    indices.append(possible_inds[0])
                else:
                    raise ValueError('Multiple indices found')

                # True if all vertices are basal
                is_basal *= self._is_basal(vertex)

            # For efficiency
            first_idx = indices[1]
            indices.append(first_idx)

            indices = self._sort_to_counterclockwise(indices, all_vertices)

            # Pad
            indices += [-1] * (self._max_vertices - len(indices))
            indices.extend([-1] * (self._max_vertices - len(indices)))

            all_indices.append(indices)
            basal_mask.append(is_basal)

        all_indices = np.array(all_indices)
        basal_mask = np.array(basal_mask)

        return all_indices, all_vertices, basal_mask

    def _get_fixed_inds(self):
        # Selection radius = 10
        ## fixed_inds =  [3, 4, 9, 10]
        # Selection radius = 20
        ## fixed_inds = [3, 4, 13, 26, 22, 19, 8, 9]
        fixed_inds = []
        return fixed_inds


class _VoronoiPolygons(_Polygons):
    def __init__(self):
        self._init_polygons = 1200
        self._all_polygon_vertex_inds, self._vertices = (
            self._make_init_polygons()
        )
        self._max_vertices = self._find_max_vertices()
        self._polygon_inds = self._finalize_polygon_inds()
        self._valid_mask = (self._polygon_inds != -1)
        self._fixed_mask = np.ones_like(self._vertices)
        self._basal_mask = np.ones(self._polygon_inds.shape[0], dtype=bool)
        self._boundary_mask = self._find_boundary_mask()

    def _extend_region(self, region):
        region.insert(0, region[-1])
        region.append(region[1])
        return region

    def _generate_random_points(self, generation_bounds):
        xs = np.random.uniform(
            generation_bounds[0], generation_bounds[1], self._init_polygons
        )
        ys = np.random.uniform(
            generation_bounds[2], generation_bounds[3], self._init_polygons
        )
        points = np.vstack((xs, ys)).T
        return points

    @staticmethod
    def _lloyd_relaxation(points, bounds):
        n_iterations = 20
        for _ in range(n_iterations):
            vor = Voronoi(points)
            new_points = []

            for i, region_idx in enumerate(vor.point_region):
                vertices = vor.regions[region_idx]
                if -1 in vertices or len(vertices) == 0:
                    new_points.append(points[i])  # fallback to original point
                    continue

                polygon = np.array([vor.vertices[j] for j in vertices])
                centroid = polygon.mean(axis=0)

                if ((bounds[0] <= centroid[0] <= bounds[1]) and
                    (bounds[2] <= centroid[1] <= bounds[3])):
                    new_points.append(centroid)
                else:
                    new_points.append(points[i])  # fallback again
            points = np.array(new_points)
        return points

    @staticmethod
    def _add_artificial_points(points, bounds):
        extra_points = [
            [bounds[0] - 1, bounds[2] - 1],
            [bounds[0] - 1, bounds[3] + 1],
            [bounds[1] + 1, bounds[2] - 1],
            [bounds[1] + 1, bounds[3] + 1]
        ]
        all_points = np.concatenate([points, extra_points])
        return all_points

    @staticmethod
    def _region_fully_inside_ellipse(vertices, center, radii):
        cx, cy = center
        rx, ry = radii
        for x, y in vertices:
            norm_x = (x - cx) / rx
            norm_y = (y - cy) / ry
            if norm_x ** 2 + norm_y ** 2 > 1:
                return False
        return True

    def _extract_vertex_indexed_cells(self, vor, ellipse_center, ellipse_radii):
        allowed_vertices = []
        vertex_index_map = {}
        all_polygon_vertex_inds = []

        n_non_artificial_points = len(vor.points) - 4
        for region_idx in vor.point_region[:n_non_artificial_points]:
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue

            raw_polygon = [vor.vertices[i] for i in region]
            if not self._region_fully_inside_ellipse(
                raw_polygon, ellipse_center, ellipse_radii
            ):
                continue

            polygon = Polygon(raw_polygon)
            if not polygon.is_valid:
                continue

            vertex_inds = []
            for vertex in raw_polygon:
                # Round to avoid floating point noise
                key = tuple(np.round(vertex, decimals=8))
                if key not in vertex_index_map:
                    vertex_index_map[key] = len(allowed_vertices)
                    allowed_vertices.append(vertex)
                vertex_inds.append(vertex_index_map[key])

            vertex_inds = self._extend_region(vertex_inds)
            vertex_inds = np.array(vertex_inds)
            vertex_inds = self._sort_to_counterclockwise(
                vertex_inds, np.array(allowed_vertices)
            )

            all_polygon_vertex_inds.append(vertex_inds)

        allowed_vertices = np.array(allowed_vertices)
        return all_polygon_vertex_inds, allowed_vertices

    @staticmethod
    def _separate_close_vertices(vertices):
        min_dist = 1.0
        jitter_scale = 1.0
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

    def _make_init_polygons(self):
        generation_bounds = 100.0 * np.array([0.0, 1.0, 0.0, 1.0])
        points = self._generate_random_points(generation_bounds)
        relaxed_points = self._lloyd_relaxation(points, generation_bounds)

        all_points = self._add_artificial_points(
            relaxed_points, generation_bounds
        )

        vor = Voronoi(all_points)

        ellipse_center = (40.0, 45.0)
        ellipse_radii = (20.0, 15.0)

        all_polygon_vertex_inds, vertices = self._extract_vertex_indexed_cells(
            vor, ellipse_center, ellipse_radii
        )

        vertices = self._separate_close_vertices(vertices)

        return all_polygon_vertex_inds, vertices

    def _find_max_vertices(self):
        max_vertices = 0
        for vertex_inds in self._all_polygon_vertex_inds:
            max_vertices = max(max_vertices, len(vertex_inds))
        return max_vertices

    def _finalize_polygon_inds(self):
        all_polygon_inds = []
        for vertex_inds in self._all_polygon_vertex_inds:
            n_padding_values = self._max_vertices - len(vertex_inds)
            padding_array = np.full((n_padding_values,), -1, dtype=np.long)
            polygon_inds = np.concatenate(
                [np.array(vertex_inds), padding_array]
            )
            all_polygon_inds.append(polygon_inds)

        all_polygon_inds = np.stack(all_polygon_inds)
        return all_polygon_inds


class _SinglePolygon(_Polygons):
    def __init__(self):
        self._n_vertices = 8
        self._vertices = self._make_init_polygons()
        self._polygon_inds = self._find_polygon_inds()
        self._valid_mask = (self._polygon_inds != -1)
        self._fixed_mask = np.array([1.0])
        self._basal_mask = np.array([True])
        self._boundary_mask = self._find_boundary_mask()

    def _make_init_polygons(self):
        if self._n_vertices < 3:
            raise ValueError('A polygon must have at least 3 vertices.')

        origin = np.array([40.0, 45.0])
        radius = 15.0

        angle_steps = np.linspace(0, 2*np.pi, self._n_vertices, endpoint=False)

        xs = origin[0] + radius * np.cos(angle_steps)
        ys = origin[1] + radius * np.sin(angle_steps)
        vertices = np.column_stack((xs, ys))

        return vertices

    def _find_polygon_inds(self):
        polygon_inds = np.arange(self._n_vertices)
        polygon_inds = np.concatenate(
            [polygon_inds, polygon_inds[:2]]
        )
        polygon_inds = self._sort_to_counterclockwise(
            polygon_inds, self._vertices
        )
        return polygon_inds.reshape(1, -1)


class _AbstractFactory(ABC):
    @abstractmethod
    def _make_params_and_polygons(self):
        pass

    def get_polygons(self):
        return self._polygons

    def get_outer_shape(self):
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
                origin = (40.0, 70.0)
                polygons = _MeshPolygons()
            case 'simple':
                a = 23.0
                b = a * 1.2
                origin = (40.0, 45.0)
                polygons = _SimpleMeshPolygons()
            case 'voronoi':
                a = 23.0
                b = a * 1.2
                origin = (40.0, 45.0)
                polygons = _VoronoiPolygons()
            case 'single':
                a = 15.0
                b = a * 1.5
                origin = (40.0, 45.0)
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
                origin = (40.0, 60.0)
                polygons = _MeshPolygons()
            case 'simple':
                a = 12.0
                polygons = _SimpleMeshPolygons()
                origin = (40.0, 23.0)
            case 'voronoi':
                a = 0.6
                origin = (0.5, 0.5)
                polygons = _VoronoiPolygons()
            case 'single':
                a = 10.0
                origin = (40.0, 45.0)
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
                origin = (40.0, 60.0)
                polygons = _MeshPolygons()
            case 'simple':
                a = 700.0
                origin = (40.0, 46.0)
                polygons = _SimpleMeshPolygons()
            case 'voronoi':
                a = 700.0
                origin = (40.0, 46.0)
                polygons = _VoronoiPolygons()
            case 'single':
                a = 700.0
                origin = (40.0, 46.0)
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

    def _gielis(self, angle, a, b, m, n1, n2, n3):
        rs = (
            np.abs(np.cos(m * angle / 4) / a)**n2 +
            np.abs(np.sin(m * angle / 4) / b)**n3
        )**(-1 / n1)
        return rs

    def _make_outer_shape(self):
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=True)
        rs = self._gielis(angles,
                          self._shape_params['a'],
                          self._shape_params['b'],
                          self._shape_params['m'],
                          self._shape_params['n1'],
                          self._shape_params['n2'],
                          self._shape_params['n3'])
        xs = rs * np.sin(angles) + self._shape_params['origin'][0]
        ys = rs * np.cos(angles) + self._shape_params['origin'][1]

        # Cheat to make it look like a petal
        scale = 0.8
        shift = 0.5 * (1 - scale) * (xs[0] + xs[-1])
        xs = scale * xs + shift

        petal = np.stack([xs, ys], axis=1)
        return petal


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
