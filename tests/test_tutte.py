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
    system, shape = request.param
    params = parameters.Params(system=system, shape=shape)
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
    pytest.param(("few", shape), id=f"few-{shape}") for shape in _SHAPES
]
_FULL_CASE = pytest.param(("full", "square"), id="full-square")


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


@pytest.mark.parametrize("mapping", _FEW_CASES + [_FULL_CASE], indirect=True)
def test_tutte_keeps_basal_vertices_on_base(mapping: _Mapping) -> None:
    basal_inds = np.flatnonzero(~mapping.polygons.free_mask[:, 1])
    base_y = init_systems.Coords.base_origin[1]
    np.testing.assert_allclose(
        mapping.mapped[basal_inds, 1], base_y, atol=1e-9
    )


@pytest.mark.parametrize("mapping", _FEW_CASES + [_FULL_CASE], indirect=True)
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


@pytest.mark.parametrize("mapping", _FEW_CASES, indirect=True)
def test_tutte_mapping_has_no_edge_crossings(mapping: _Mapping) -> None:
    poly_idx_lists = init_systems.make_poly_idx_lists(mapping.polygons.indices)
    n_crossings = metrics.count_edge_crossings(mapping.mapped, poly_idx_lists)
    assert n_crossings == 0


@pytest.mark.parametrize("mapping", _FEW_CASES, indirect=True)
def test_tutte_mapping_does_not_fold_cells(mapping: _Mapping) -> None:
    cells = init_systems.get_shapely_polygons(
        mapping.mapped, mapping.polygons.indices
    )
    # A folded cell either intersects itself or turns clockwise
    assert all(
        cell.is_valid and shapely.is_ccw(cell.exterior) for cell in cells
    )
