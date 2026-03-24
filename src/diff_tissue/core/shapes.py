from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

from .jax_bootstrap import jnp, struct
from . import init_systems


class _Shape(ABC):
    def __init__(self, mesh_area, vertex_numbers):
        self._mesh_area = mesh_area
        self._vertex_numbers = vertex_numbers

    @property
    @abstractmethod
    def _raw_shape(self):
        pass

    @staticmethod
    def _resample_curve(points, n_points, spacing=None):
        points = np.asarray(points)
        diffs = np.diff(points, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        cumulative_lengths = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        total_length = cumulative_lengths[-1]

        if spacing is not None:
            n_points = int(np.floor(total_length / spacing)) + 1
        elif n_points is None:
            raise ValueError("You must provide either num_points or spacing.")

        target_lengths = np.linspace(0, total_length, n_points)

        resampled = []
        for t in target_lengths:
            idx = np.searchsorted(cumulative_lengths, t) - 1
            idx = min(max(idx, 0), len(points) - 2)
            t0, t1 = cumulative_lengths[idx], cumulative_lengths[idx + 1]
            p0, p1 = points[idx], points[idx + 1]
            # Handle zero-length segments
            if t1 == t0:
                resampled.append(p0)
            else:
                alpha = (t - t0) / (t1 - t0)
                resampled.append((1 - alpha) * p0 + alpha * p1)
        return np.array(resampled)

    def _make_non_basal_vertices(self, xs, ys):
        vertices = np.array([(x, y) for x, y in zip(xs, ys)])
        vertices = self._resample_curve(
            vertices, self._vertex_numbers.non_basal
        )
        return vertices

    def _make_basal_vertices(self, lower_r):
        basal_xs = np.linspace(-lower_r, lower_r, self._vertex_numbers.basal)[
            1:-1
        ]
        basal_vertices = np.array([(x, 0.0) for x in basal_xs])
        return basal_vertices

    def _construct_target_boundary(self, non_basal_xs, non_basal_ys, lower_r):
        non_basal_vertices = self._make_non_basal_vertices(
            non_basal_xs, non_basal_ys
        )
        basal_vertices = self._make_basal_vertices(lower_r)
        target_boundary = np.concatenate(
            [non_basal_vertices, basal_vertices], axis=0
        )
        return target_boundary

    @staticmethod
    def _calc_shape_area(polygon):
        """Calculate the area of a polygon, using the shoelace formula.

        Assumes no repeating vertices.

        Returns:
            float: Area of polygon.
        """
        xs = polygon[:, 0]
        ys = polygon[:, 1]

        area = 0.5 * abs(
            np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
        )
        return area

    @staticmethod
    def _calc_scale(mesh_area, raw_shape_area):
        scale = np.sqrt(mesh_area / raw_shape_area)
        return scale

    def _transform_raw_shape(self):
        raw_shape_area = self._calc_shape_area(self._raw_shape)
        scale = self._calc_scale(self._mesh_area, raw_shape_area)
        target_boundary = scale * self._raw_shape
        target_boundary += init_systems.Coords.base_origin
        return target_boundary

    def _validate_rescaled_area(self, shape_area):
        if not np.isclose(self._mesh_area - shape_area, 0.0):
            raise ValueError("System and mesh areas do not match!")

    @cached_property
    def vertices(self):
        target_boundary = self._transform_raw_shape()
        shape_area = self._calc_shape_area(target_boundary)
        self._validate_rescaled_area(shape_area)
        return target_boundary

    @cached_property
    def segments(self):
        segments_ = init_systems.get_segments(self.vertices)
        return segments_


class _NonConvexShape(_Shape):
    def __init__(self, mesh_area, vertex_numbers):
        super().__init__(mesh_area, vertex_numbers)
        self._height = 3.0
        self._lower_r = 1.5

    @cached_property
    def _raw_shape(self):
        non_basal_xs = self._lower_r * np.array([1.0, 1.4, 1.8, 1.9, 1.2, 0.4])
        non_basal_xs = np.concatenate([non_basal_xs, np.flip(-non_basal_xs)])
        non_basal_ys = self._height * np.array([0.0, 0.3, 0.7, 1.1, 1.2, 1.0])
        non_basal_ys = np.concatenate([non_basal_ys, np.flip(non_basal_ys)])
        target_boundary = self._construct_target_boundary(
            non_basal_xs, non_basal_ys, self._lower_r
        )
        return target_boundary


class _Petal(_Shape):
    def __init__(self, mesh_area, vertex_numbers):
        super().__init__(mesh_area, vertex_numbers)
        self._lower_r: float
        self._height: float
        self._stretch_strength: float

    @cached_property
    def _raw_shape(self):
        xs = np.linspace(
            -self._lower_r, self._lower_r, self._vertex_numbers.non_basal
        )
        non_basal_ys = self._height * np.sqrt(1 - (xs / self._lower_r) ** 2)

        factor = 1 + self._stretch_strength * non_basal_ys / self._height
        non_basal_xs = xs * factor

        target_boundary = self._construct_target_boundary(
            non_basal_xs, non_basal_ys, self._lower_r
        )
        return target_boundary


class _ShortPetal(_Petal):
    def __init__(self, mesh_area, vertex_numbers):
        super().__init__(mesh_area, vertex_numbers)
        self._lower_r = 20.0
        self._height = 60.0
        self._stretch_strength = 2.0


class _LongPetal(_Petal):
    def __init__(self, mesh_area, vertex_numbers):
        super().__init__(mesh_area, vertex_numbers)
        self._lower_r = 20.0
        self._height = 100.0
        self._stretch_strength = 3.0


class IsoTrapezoid(_Shape):
    def __init__(self, mesh_area, vertex_numbers, angle):
        super().__init__(mesh_area, vertex_numbers)
        self._angle = angle  # ccw beetween base and right leg (degrees)
        self._angle_rad = self._angle_to_rads()
        self._side_length = 1.0
        self._lower_r = self._side_length / 2

    @staticmethod
    def _validate_angle(angle):
        if not (0 < angle <= 120):
            raise ValueError("Angle must be between 0 and 120 degrees.")
        return angle

    def _angle_to_rads(self):
        valid_angle = self._validate_angle(self._angle)
        return np.radians(valid_angle)

    @cached_property
    def _upper_r(self):
        return self._lower_r + self._side_length * np.cos(self._angle_rad)

    @cached_property
    def _height(self):
        return self._side_length * np.sin(self._angle_rad)

    @cached_property
    def _raw_shape(self):
        non_basal_xs = np.array(
            [self._lower_r, self._upper_r, -self._upper_r, -self._lower_r]
        )
        non_basal_ys = np.array([0.0, self._height, self._height, 0.0])

        target_boundary = self._construct_target_boundary(
            non_basal_xs, non_basal_ys, self._lower_r
        )
        return target_boundary


def get_target_boundary(shape, mesh_area, vertex_numbers):
    match shape:
        case "nconv":
            shape = _NonConvexShape(mesh_area, vertex_numbers)
        case "petal":
            shape = _ShortPetal(mesh_area, vertex_numbers)
        case "long_petal":
            shape = _LongPetal(mesh_area, vertex_numbers)
        case "trapezoid":
            shape = IsoTrapezoid(mesh_area, vertex_numbers, angle=81.87)
        case "narrow":
            shape = IsoTrapezoid(mesh_area, vertex_numbers, angle=110.0)
        case "square":
            shape = IsoTrapezoid(mesh_area, vertex_numbers, angle=90.0)
        case "wide":
            shape = IsoTrapezoid(mesh_area, vertex_numbers, angle=70.0)
        case "triangle":
            shape = IsoTrapezoid(mesh_area, vertex_numbers, angle=120.0)
        case _:
            raise ValueError("Invalid target boundary shape!")
    return shape


@struct.dataclass
class JaxTargetBoundary:
    vertices: jnp.ndarray
    segments: jnp.ndarray


def get_jax_target_boundary(polygons, params):
    target_boundary = get_target_boundary(
        params.shape, polygons.mesh_area, init_systems.VertexNumbers(polygons)
    )
    jax_target_boundary = JaxTargetBoundary(
        vertices=jnp.array(target_boundary.vertices),
        segments=jnp.array(target_boundary.segments),
    )
    return jax_target_boundary
