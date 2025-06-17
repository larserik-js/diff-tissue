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
    full_mesh_origin = np.array([40.0, 0.0])
    full_mesh_base = np.array([40.0, 18.635])
    shape_origin = full_mesh_origin + (0.0, 45.0)


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
        dist_from_origin = np.linalg.norm(vertex - Coords.full_mesh_origin)
        # Basal radius of 45.0 is good for Hibiscus trionum
        # (using full mesh)
        basal_radius = np.inf
        return dist_from_origin < basal_radius

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


class _VoronoiPolygons(_Polygons):
    def __init__(self):
        self._n_polygons = 200
        self._generating_shape = self._get_generating_shape()
        self._all_polygon_inds, self._vertices = (
            self._make_init_polygons()
        )
        self._max_vertices = self._find_max_vertices()
        self._polygon_inds = self._finalize_polygon_inds()
        self._valid_mask = (self._polygon_inds != -1)
        self._fixed_mask = np.ones_like(self._vertices)
        self._basal_mask = np.ones(self._polygon_inds.shape[0], dtype=bool)
        self._boundary_mask = self._find_boundary_mask()

    def _get_generating_shape(self):
        num_points = 20
        thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        cx, cy = Coords.shape_origin
        rx, ry = 15.0, 20.0
        xs = cx + rx * np.cos(thetas)
        ys = cy + ry * np.sin(thetas)
        coords = [point for point in zip(xs, ys)]

        box = Polygon(coords)
        return box

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

    def _make_clipped_voronoi(self, relaxed_points, radius=None):
        vor = Voronoi(relaxed_points)
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        all_polygon_inds = []
        new_vertices = vor.vertices.tolist()

        centroid = vor.points.mean(axis=0)
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
                        np.dot(midpoint - centroid, normal)
                    ) * normal
                    far_point = vor.vertices[v2] + direction * radius

                    new_poly_inds.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())

                # Sort region counterclockwise
                vertices = np.array([new_vertices[idx] for idx in new_poly_inds])
                c = vertices.mean(axis=0)
                angles = np.arctan2(vertices[:,1] - c[1], vertices[:,0] - c[0])
                new_poly_inds = np.array(new_poly_inds)[np.argsort(angles)]
                all_polygon_inds.append(new_poly_inds.tolist())

        all_vertices = np.array(new_vertices)

        clipped_polygons = self._clip_polygons(all_polygon_inds, all_vertices)

        return clipped_polygons

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

    def _extend(self, polygon):
        polygon.insert(0, polygon[-1])
        polygon.append(polygon[1])
        return polygon

    def _extend_polygons(self, polygons):
        extended_polygons = []
        for polygon in polygons:
            extended_polygon = self._extend(polygon)
            extended_polygons.append(extended_polygon)
        return extended_polygons

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
        relaxed_points = self._generate_relaxed_random_points()
        polygons = self._make_clipped_voronoi(relaxed_points)

        all_polygon_inds, vertices = self._make_shared_vertex_structure(
            polygons
        )

        all_polygon_inds = self._extend_polygons(all_polygon_inds)
        vertices = self._separate_close_vertices(vertices)

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

        radius = 15.0

        angle_steps = np.linspace(0, 2*np.pi, self._n_vertices, endpoint=False)

        xs = Coords.shape_origin[0] + radius * np.cos(angle_steps)
        ys = Coords.shape_origin[1] + radius * np.sin(angle_steps)
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
