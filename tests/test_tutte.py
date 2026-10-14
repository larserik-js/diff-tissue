from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import shapely

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, metrics, shapes, tutte


def test_cells_to_edges_removes_closing_duplicates() -> None:
    edges = tutte._cells_to_edges([[0, 1, 2, 0], [1, 3, 2]])
    assert sorted(edges) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]


def test_best_cyclic_shift() -> None:
    circle = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    shifted = np.roll(circle, 1, axis=0)
    assert tutte._best_cyclic_shift(circle, shifted) == 1


def test_rotate_rows() -> None:
    arr = np.arange(8).reshape(4, 2)
    np.testing.assert_array_equal(
        tutte._rotate_rows(arr, 5), np.roll(arr, -1, axis=0)
    )


def test_get_bottom_right_vertex_ignores_vertices_above_base() -> None:
    # The side bulges out to the right just above the base corner at (1, 0)
    vertices = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [1.5, 0.5], [0.0, 2.0], [-1.5, 0.5]]
    )
    np.testing.assert_array_equal(
        tutte._get_bottom_right_vertex(vertices), [1.0, 0.0]
    )


@dataclass
class _Fan:
    """A center vertex surrounded by a square of four boundary vertices,
    ordered counter-clockwise from the positive x-axis."""

    vertices: npt.NDArray[np.floating]
    cells: list[list[int]]
    boundary_inds: npt.NDArray[np.integer]
    center_idx: int


def _make_fan() -> _Fan:
    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    cells = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
    return _Fan(vertices, cells, np.array([1, 2, 3, 4]), center_idx=0)


# The fan's boundary, scaled and translated
_FAN_SCALE = 2.0
_FAN_OFFSET = np.array([3.0, 1.0])


def _fan_target(fan: _Fan) -> npt.NDArray[np.floating]:
    return _FAN_SCALE * fan.vertices[fan.boundary_inds] + _FAN_OFFSET


def test_circle_embedding_keeps_center_at_origin() -> None:
    fan = _make_fan()
    uv_init, _, _ = tutte._map_to_given_shape(
        fan.vertices, fan.cells, fan.boundary_inds, _fan_target(fan)
    )
    np.testing.assert_allclose(uv_init[fan.center_idx], 0.0, atol=1e-12)


def test_map_to_given_shape_places_boundary_on_target() -> None:
    fan = _make_fan()
    target = _fan_target(fan)
    _, mapped, _ = tutte._map_to_given_shape(
        fan.vertices, fan.cells, fan.boundary_inds, target
    )
    np.testing.assert_allclose(mapped[fan.boundary_inds], target)


def test_map_to_given_shape_maps_center_to_target_center() -> None:
    fan = _make_fan()
    _, mapped, _ = tutte._map_to_given_shape(
        fan.vertices, fan.cells, fan.boundary_inds, _fan_target(fan)
    )
    np.testing.assert_allclose(mapped[fan.center_idx], _FAN_OFFSET)


def test_map_to_given_shape_auto_align_undoes_cyclic_shift() -> None:
    fan = _make_fan()
    target = _fan_target(fan)
    shifted_target = np.roll(target, 1, axis=0)

    _, mapped, _ = tutte._map_to_given_shape(
        fan.vertices,
        fan.cells,
        fan.boundary_inds,
        shifted_target,
        auto_align=True,
    )
    np.testing.assert_allclose(mapped[fan.boundary_inds], target)


def test_map_to_given_shape_length_mismatch() -> None:
    fan = _make_fan()
    with pytest.raises(ValueError, match="boundary_target length"):
        tutte._map_to_given_shape(
            fan.vertices, fan.cells, fan.boundary_inds, fan.vertices[:2]
        )


def test_get_mapped_vertices_keeps_target_ring_order() -> None:
    # A fan of eight triangles around a center vertex in a square
    center = [[1.0, 1.0]]
    ring = [[2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1], [0, 0], [1, 0]]
    vertices = np.array(center + ring, dtype=float)
    boundary_inds = np.arange(1, 9)
    # Each cell is stored as [last, v0, v1, v2, v0]
    indices = np.array(
        [
            [b, 0, a, b, 0]
            for a, b in zip(boundary_inds, np.roll(boundary_inds, -1))
        ]
    )
    # A C shape that opens to the right. It is not star-shaped around the mean
    # of its vertices, so sorting its vertices by angle scrambles the ring
    target = np.array(
        [[3, 0], [3, 1], [1, 1], [1, 2], [3, 2], [3, 3], [0, 3], [0, 0]],
        dtype=float,
    )
    mapped = tutte.get_mapped_vertices(
        vertices, indices, boundary_inds, target
    )
    np.testing.assert_allclose(mapped[boundary_inds], target, atol=1e-12)


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


@dataclass
class _Mapping:
    polygons: init_systems._Polygons
    target_vertices: npt.NDArray[np.floating]
    mapped: npt.NDArray[np.floating]


@pytest.fixture(scope="module")
def mapping(request: pytest.FixtureRequest) -> _Mapping:
    system, shape, seed = request.param
    params = parameters.Params(system=system, shape=shape, seed=seed)
    polygons = init_systems.get_system(params)
    target_vertices = shapes.get_target_boundary(params, polygons).vertices
    mapped = tutte.get_mapped_vertices(
        polygons.init_vertices,
        polygons.indices,
        polygons.boundary_inds,
        target_vertices,
    )
    return _Mapping(polygons, target_vertices, mapped)


_FEW_CASES = [
    pytest.param(("few", shape, 0), id=f"few-{shape}") for shape in _SHAPES
]
_MANY_CASES = [
    pytest.param(("many", shape, 0), id=f"many-{shape}") for shape in _SHAPES
]
_FULL_CASE = pytest.param(("full", "square", 0), id="full-square")


_CONCAVE_DIP_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Tutte embedding is only guaranteed for convex targets; "
    + "cells fold across the concave dip in the top edge",
)
# Known complex_nconv failures. In "few" seed 4 the cells fold, but no
# interior vertex leaves the target
_DIP_CASES_OUTSIDE = [
    pytest.param(
        (system, "complex_nconv", seed),
        id=f"{system}-complex_nconv-seed{seed}",
        marks=_CONCAVE_DIP_XFAIL,
    )
    for system, seed in [("few", 2), ("many", 10)]
]
_DIP_CASES_FOLDED = _DIP_CASES_OUTSIDE + [
    pytest.param(
        ("few", "complex_nconv", 4),
        id="few-complex_nconv-seed4",
        marks=_CONCAVE_DIP_XFAIL,
    )
]


@pytest.mark.parametrize("mapping", _FEW_CASES + [_FULL_CASE], indirect=True)
def test_tutte_maps_boundary_one_to_one_onto_target(
    mapping: _Mapping,
) -> None:
    boundary_vertices = mapping.mapped[mapping.polygons.boundary_inds]
    dists = np.linalg.norm(
        boundary_vertices[:, None] - mapping.target_vertices[None], axis=2
    )
    np.testing.assert_allclose(dists.min(axis=1), 0.0, atol=1e-9)
    hit_target_inds = np.unique(dists.argmin(axis=1))
    assert len(hit_target_inds) == len(mapping.target_vertices)


@pytest.mark.parametrize(
    "mapping", _FEW_CASES + _MANY_CASES + [_FULL_CASE], indirect=True
)
def test_tutte_keeps_basal_vertices_on_base(mapping: _Mapping) -> None:
    basal_inds = np.flatnonzero(~mapping.polygons.free_mask[:, 1])
    base_y = init_systems.Coords.base_origin[1]
    np.testing.assert_allclose(
        mapping.mapped[basal_inds, 1], base_y, atol=1e-9
    )


@pytest.mark.parametrize(
    "mapping", _FEW_CASES + [_FULL_CASE] + _DIP_CASES_OUTSIDE, indirect=True
)
def test_tutte_keeps_interior_vertices_inside_target(
    mapping: _Mapping,
) -> None:
    n_vertices = len(mapping.mapped)
    interior_inds = np.setdiff1d(
        np.arange(n_vertices), mapping.polygons.boundary_inds
    )
    target_polygon = shapely.Polygon(mapping.target_vertices)
    interior_points = shapely.points(mapping.mapped[interior_inds])
    assert np.all(shapely.covers(target_polygon, interior_points))


@pytest.mark.parametrize(
    "mapping", _FEW_CASES + [_FULL_CASE] + _DIP_CASES_FOLDED, indirect=True
)
def test_tutte_mapping_has_no_edge_crossings(mapping: _Mapping) -> None:
    poly_idx_lists = init_systems.make_poly_idx_lists(mapping.polygons.indices)
    n_crossings = metrics.count_edge_crossings(mapping.mapped, poly_idx_lists)
    assert n_crossings == 0


@pytest.mark.parametrize(
    "mapping", _FEW_CASES + [_FULL_CASE] + _DIP_CASES_FOLDED, indirect=True
)
def test_tutte_mapping_does_not_fold_cells(mapping: _Mapping) -> None:
    cells = init_systems.get_shapely_polygons(
        mapping.mapped, mapping.polygons.indices
    )
    # A folded cell either intersects itself or turns clockwise
    assert all(
        cell.is_valid and shapely.is_ccw(cell.exterior) for cell in cells
    )
