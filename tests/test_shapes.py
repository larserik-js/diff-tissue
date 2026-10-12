from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial import KDTree
import shapely

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, shapes


def _count_pairs_of_duplicates(
    points: npt.NDArray[np.floating], tol: float = 1e-6
) -> int:
    tree = KDTree(points)
    pairs = tree.query_pairs(r=tol)
    return len(pairs)


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        pytest.param([[0.0, 0.0], [1.0, 0.0]], 0, id="distinct"),
        pytest.param([[0.0, 0.0], [0.0, 0.0]], 1, id="one_pair"),
        pytest.param(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            2,
            id="two_pairs",
        ),
        pytest.param([[0.0, 0.0], [0.0, 1e-8]], 1, id="within_tolerance"),
    ],
)
def test_duplicate_counter(points: list[list[float]], expected: int) -> None:
    assert _count_pairs_of_duplicates(np.array(points)) == expected


_SHAPES = [
    "petal",
    "long_petal",
    "trapezoid",
    "narrow_trapezoid",
    "wide_trapezoid",
    "square",
    "nconv",
    "complex_nconv",
]
# Limits of the valid trapezoid angles, which the shapes above do not cover
_EXTREME_TRAPEZOID_ANGLES = [61.0, 120.0]


@dataclass
class _Target:
    polygons: init_systems._Polygons
    boundary: shapes._Shape

    @property
    def vertices(self) -> npt.NDArray[np.floating]:
        return self.boundary.vertices


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(parameters.Params(shape=shape), id=shape)
        for shape in _SHAPES
    ]
    + [
        pytest.param(
            parameters.Params(shape="trapezoid", trapezoid_angle=angle),
            id=f"trapezoid_{angle}",
        )
        for angle in _EXTREME_TRAPEZOID_ANGLES
    ],
)
def target(request: pytest.FixtureRequest) -> _Target:
    params = request.param
    polygons = init_systems.get_system(params)
    boundary = shapes.get_target_boundary(params, polygons)
    return _Target(polygons, boundary)


def test_target_boundary_has_no_duplicate_points(target: _Target) -> None:
    assert _count_pairs_of_duplicates(target.vertices) == 0


def test_target_boundary_has_one_vertex_per_mesh_boundary_vertex(
    target: _Target,
) -> None:
    vertex_numbers = init_systems.VertexNumbers(target.polygons)
    assert len(target.vertices) == vertex_numbers.boundary


@pytest.mark.parametrize(
    "vertex_set", ["vertices", "smooth_vertices", "reduced_vertices"]
)
def test_target_boundary_area_matches_mesh(
    target: _Target, vertex_set: str
) -> None:
    vertices = getattr(target.boundary, vertex_set)
    area = shapely.Polygon(vertices).area
    assert area == pytest.approx(target.polygons.mesh_area)


def test_target_boundary_is_simple_polygon(target: _Target) -> None:
    assert shapely.Polygon(target.vertices).is_valid


def test_target_boundary_has_basal_vertices_on_base(target: _Target) -> None:
    base_y = init_systems.Coords.base_origin[1]
    on_base = np.isclose(target.vertices[:, 1], base_y)
    vertex_numbers = init_systems.VertexNumbers(target.polygons)
    assert np.count_nonzero(on_base) == vertex_numbers.basal


def test_target_boundary_is_above_base(target: _Target) -> None:
    base_y = init_systems.Coords.base_origin[1]
    assert np.min(target.vertices[:, 1]) >= base_y - 1e-9


def test_target_boundary_is_mirror_symmetric(target: _Target) -> None:
    mid_x = init_systems.Coords.base_origin[0]
    mirrored = target.vertices.copy()
    mirrored[:, 0] = 2 * mid_x - mirrored[:, 0]

    dists, _ = KDTree(target.vertices).query(mirrored)
    np.testing.assert_allclose(dists, 0.0, atol=1e-9)


def test_get_target_boundary_invalid_shape() -> None:
    params = parameters.Params(shape="invalid")
    polygons = init_systems.get_system(params)
    with pytest.raises(ValueError, match="Invalid target boundary shape"):
        shapes.get_target_boundary(params, polygons)


@pytest.mark.parametrize("angle", [60.0, 120.1])
def test_trapezoid_invalid_angle(angle: float) -> None:
    params = parameters.Params(shape="trapezoid", trapezoid_angle=angle)
    polygons = init_systems.get_system(params)
    with pytest.raises(ValueError, match="Angle must be"):
        shapes.get_target_boundary(params, polygons)


def test_get_jax_target_boundary() -> None:
    params = parameters.Params()
    polygons = init_systems.get_system(params)
    target_boundary = shapes.get_target_boundary(params, polygons)
    jax_target_boundary = shapes.get_jax_target_boundary(polygons, params)

    np.testing.assert_allclose(
        jax_target_boundary.vertices, target_boundary.vertices
    )
    np.testing.assert_allclose(
        jax_target_boundary.segments, target_boundary.segments
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({"n_points": 5}, [0.0, 0.5, 1.0, 1.5, 2.0], id="n"),
        pytest.param(
            {"n_points": None, "spacing": 1.0}, [0.0, 1.0, 2.0], id="spacing"
        ),
    ],
)
def test_resample_curve(
    kwargs: dict[str, float | None], expected: list[float]
) -> None:
    # Starts with a zero-length segment
    points = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    resampled = shapes._resample_curve(points, **kwargs)
    np.testing.assert_allclose(resampled[:, 0], expected)
    np.testing.assert_allclose(resampled[:, 1], 0.0)


def test_resample_curve_requires_n_points_or_spacing() -> None:
    with pytest.raises(ValueError, match="num_points or spacing"):
        shapes._resample_curve([[0.0, 0.0], [1.0, 0.0]], n_points=None)
