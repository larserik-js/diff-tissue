import json
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi


class _Polygons:
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
                    print(f'Interior edge: {sorted_edge}')

                all_edges.add(sorted_edge)

        boundary_edges = all_edges - interior_edges
        boundary_inds = np.array(list(boundary_edges)).flatten()
        boundary_inds = np.unique(boundary_inds)
        boundary_mask = np.zeros_like(self._vertices, dtype=bool)
        boundary_mask[boundary_inds] = True

        return boundary_mask

    def get_polygon_inds(self):
        return self._polygon_inds

    def get_valid_mask(self):
        return self._valid_mask

    def get_vertices(self):
        return self._vertices

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
        return dist_from_pseudo_origin < 50.0

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
        self._selection_radius = 10.0
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
            # Pad
            indices += [-1] * (self._max_vertices - len(indices))
            indices.extend([-1] * (self._max_vertices - len(indices)))

            all_indices.append(indices)
            basal_mask.append(is_basal)

        all_indices = np.array(all_indices)
        basal_mask = np.array(basal_mask)

        return all_indices, all_vertices, basal_mask

    def _get_fixed_inds(self):
        fixed_inds =  [3, 4, 9, 10]
        return fixed_inds


class _VoronoiPolygons(_Polygons):
    def __init__(self):
        self._init_polygons = 100
        self._all_polygon_vertex_inds, self._vertices = (
            self._make_init_polygons()
        )
        self._max_vertices = self._find_max_vertices()
        self._polygon_inds = self._finalize_polygon_inds()
        self._valid_mask = (self._polygon_inds != -1)
        self._fixed_mask = np.ones_like(self._vertices)
        self._basal_mask = np.ones(self._polygon_inds.shape[0], dtype=bool)
        self._boundary_mask = self._find_boundary_mask()

    def _is_finite(self, region):
        return (-1 not in region) and (len(region) > 0)

    def _inside_unit_square(self, vertices):
        inside_unit_square = (
            (vertices >= 0).all(axis=1) & (vertices <= 1).all(axis=1)
        )
        return inside_unit_square

    def _any_vertex_outside_unit_square(self, vertices):
        return np.any((vertices < 0) | (vertices > 1))

    def _extend_region(self, region):
        region.insert(0, region[-1])
        region.append(region[1])
        return region

    def _make_init_polygons(self):
        circumcenters = np.random.rand(self._init_polygons, 2)
        vor = Voronoi(circumcenters)
        inside_unit_square = self._inside_unit_square(vor.vertices)
        not_allowed_vertex_inds = np.where(~inside_unit_square)[0]
        allowed_vertices = vor.vertices[inside_unit_square]

        all_polygon_vertex_inds = []
        for region in vor.regions:
            if self._is_finite(region):
                vertices = vor.vertices[region]
                if self._any_vertex_outside_unit_square(vertices):
                    continue

                # For efficiency
                region = self._extend_region(region)
                vertex_inds = np.array(region)
                adjustment_inds = np.zeros_like(vertex_inds)
                for i in not_allowed_vertex_inds:
                    adjustment_inds -= (vertex_inds >= i).astype(int)
                vertex_inds += adjustment_inds
                all_polygon_vertex_inds.append(vertex_inds)

        return all_polygon_vertex_inds, allowed_vertices

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


def get_polygons(args):
    match args.init_system:
        case 'full':
            polygons = _MeshPolygons()
        case 'simple':
            polygons = _SimpleMeshPolygons()
        case 'voronoi':
            polygons = _VoronoiPolygons()
        case _:
            raise ValueError('Invalid initial configuration')
    return polygons
