from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

from . import init_systems


class _Shape(ABC):
    def __init__(self, mesh_area, vertex_numbers):
        self._build()

        self._mesh_area = mesh_area
        self._vertex_numbers = vertex_numbers
        self._raw_shape = self._make_raw_shape()

    @abstractmethod
    def _build(self):
        pass

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
            vertices, self._vertex_numbers.non_basal
        )
        return vertices

    def _make_basal_vertices(self, lower_r):
        basal_xs = np.linspace(
            -lower_r, lower_r, self._vertex_numbers.basal
        )[1:-1]
        basal_vertices = np.array([(x, 0.0) for x in basal_xs])
        return basal_vertices

    def _construct_outer_shape(self, non_basal_xs, non_basal_ys, lower_r):
        non_basal_vertices = self._make_non_basal_vertices(
            non_basal_xs, non_basal_ys
        )
        basal_vertices = self._make_basal_vertices(lower_r)
        outer_shape = np.concatenate(
            [non_basal_vertices, basal_vertices], axis=0
        )
        return outer_shape

    @abstractmethod
    def _make_raw_shape(self):
        pass

    @staticmethod
    def _calc_shape_area(polygon):
        """Calculate the area of a polygon, using the shoelace formula.

        Assumes no repeating vertices.

        Returns:
            float: Area of polygon.
        """
        xs = polygon[:, 0]
        ys = polygon[:, 1]

        area = (
            0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
        )
        return area

    @staticmethod
    def _calc_scale(mesh_area, raw_shape_area):
        scale = np.sqrt(mesh_area / raw_shape_area)
        return scale

    def _transform_raw_shape(self):
        raw_shape_area = self._calc_shape_area(self._raw_shape)
        scale = self._calc_scale(self._mesh_area, raw_shape_area)
        outer_shape = scale * self._raw_shape
        outer_shape += init_systems.Coords.base_origin
        return outer_shape

    def _validate_rescaled_area(self, shape_area):
        if not np.isclose(self._mesh_area - shape_area, 0.0):
            raise ValueError('System and mesh areas do not match!')

    @cached_property
    def vertices(self):
        outer_shape = self._transform_raw_shape()
        shape_area = self._calc_shape_area(outer_shape)
        self._validate_rescaled_area(shape_area)
        return outer_shape

    @cached_property
    def segments(self):
        segments_ = init_systems.get_segments(self.vertices)
        return segments_


class _NonConvexShape(_Shape):
    def __init__(self, mesh_area, vertex_numbers):
        super().__init__(mesh_area, vertex_numbers)

    def _build(self):
        self._height = 3.0
        self._lower_r = 1.5

    def _make_raw_shape(self):
        non_basal_xs = self._lower_r * np.array([1.0, 1.4, 1.8, 1.9, 1.2, 0.4])
        non_basal_xs = np.concatenate([non_basal_xs, np.flip(-non_basal_xs)])
        non_basal_ys = self._height * np.array([0.0, 0.3, 0.7, 1.1, 1.2, 1.0])
        non_basal_ys = np.concatenate([
            non_basal_ys, np.flip(non_basal_ys)]
        )
        outer_shape = self._construct_outer_shape(
            non_basal_xs, non_basal_ys, self._lower_r
        )
        return outer_shape


class _Petal(_Shape):
    def __init__(self, mesh_area, vertex_numbers):
        super().__init__(mesh_area, vertex_numbers)

    def _build(self):
        self._lower_r = 20.0
        self._height = 60.0
        self._stretch_strength = 2.0

    def _make_raw_shape(self):
        xs = np.linspace(
            -self._lower_r, self._lower_r, self._vertex_numbers.non_basal
        )
        non_basal_ys = (
            self._height * np.sqrt(1 - (xs / self._lower_r)**2)
        )

        factor = (
            1 + self._stretch_strength * non_basal_ys / self._height
        )
        non_basal_xs = xs * factor

        outer_shape = self._construct_outer_shape(
            non_basal_xs, non_basal_ys, self._lower_r
        )
        return outer_shape


class _Trapzeoid(_Shape):
    def __init__(self, mesh_area, vertex_numbers):
        super().__init__(mesh_area, vertex_numbers)

    def _build(self):
        self._height = 3.5
        self._lower_r = 1.5
        self._upper_r = 2.0

    def _make_raw_shape(self):
        non_basal_xs = np.array(
            [self._lower_r, self._upper_r, -self._upper_r, -self._lower_r]
        )
        non_basal_ys = np.array([0.0, self._height, self._height, 0.0])

        outer_shape = self._construct_outer_shape(
            non_basal_xs, non_basal_ys, self._lower_r
        )
        return outer_shape


class _Triangle(_Shape):
    def __init__(self, mesh_area, vertex_numbers):
        super().__init__(mesh_area, vertex_numbers)

    def _build(self):
        self._height = 2.5
        self._lower_r = 1.5

    def _make_raw_shape(self):
        non_basal_xs = np.array([self._lower_r, 0.0, -self._lower_r])
        non_basal_ys = np.array([0.0, self._height, 0.0])

        outer_shape = self._construct_outer_shape(
            non_basal_xs, non_basal_ys, self._lower_r
        )
        return outer_shape


def get_outer_shape(shape, mesh_area, vertex_numbers):
    match shape:
        case 'nconv':
            shape = _NonConvexShape(mesh_area, vertex_numbers)
        case 'petal':
            shape = _Petal(mesh_area, vertex_numbers)
        case 'trapezoid':
            shape = _Trapzeoid(mesh_area, vertex_numbers)
        case 'triangle':
            shape = _Triangle(mesh_area, vertex_numbers)
        case _:
            raise ValueError('Invalid outer shape!')
    return shape
