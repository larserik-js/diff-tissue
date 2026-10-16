import numpy as np
import pytest
import shapely

from diff_tissue.app import parameters
from diff_tissue.core import init_systems


_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_sort_counterclockwise() -> None:
    shuffled_inds = [2, 0, 3, 1]
    sorted_inds = init_systems.sort_counterclockwise(
        shuffled_inds, _UNIT_SQUARE[shuffled_inds]
    )
    # Starts at the vertex with the smallest angle around the centroid
    assert sorted_inds == [0, 1, 2, 3]


def test_get_segments_closes_polygon() -> None:
    segments = init_systems.get_segments(_UNIT_SQUARE)
    expected = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    assert segments == pytest.approx(np.array(expected))


@pytest.mark.parametrize("system", ["few", "many", "full", "single"])
def test_get_system_is_consistent(system: str) -> None:
    polygons = init_systems.get_system(parameters.Params(system=system))

    n_vertices = polygons.init_vertices.shape[0]
    assert polygons.init_vertices.shape == (n_vertices, 2)
    assert polygons.free_mask.shape == (n_vertices, 2)
    assert polygons.valid_mask.shape == polygons.indices.shape
    assert polygons.indices.max() < n_vertices
    assert polygons.mesh_area > 0.0

    assert len(polygons.boundary_inds) > 0
    assert polygons.boundary_inds.max() < n_vertices
    assert polygons.vertex_polygons.shape[0] == n_vertices
    assert polygons.poly_neighbors.shape[0] == len(polygons.indices)

    # Extended polygons: first index repeats the last, last repeats the second
    for polygon_inds in polygons.indices:
        inds = polygon_inds[polygon_inds != -1]
        assert inds[0] == inds[-2] and inds[-1] == inds[1]

    # Some vertices are fixed, but not all
    assert 0 < np.count_nonzero(~polygons.free_mask) < polygons.free_mask.size


def _assert_ccw_ring_of_mesh_edges(
    polygons: init_systems._Polygons, ring: list[int]
) -> None:
    mesh_edges: set[frozenset[int]] = set()
    for polygon_inds in polygons.indices:
        cell = polygon_inds[polygon_inds != -1][1:-1].tolist()
        mesh_edges.update(map(frozenset, zip(cell, cell[1:] + cell[:1])))

    ring_edges = map(frozenset, zip(ring, ring[1:] + ring[:1]))
    assert len(set(ring)) == len(ring)
    assert all(edge in mesh_edges for edge in ring_edges)
    boundary = shapely.LinearRing(polygons.init_vertices[ring])
    assert boundary.is_valid and shapely.is_ccw(boundary)


@pytest.mark.parametrize("system", ["few", "many", "full", "single"])
def test_boundary_inds_form_ccw_ring_of_mesh_edges(system: str) -> None:
    polygons = init_systems.get_system(parameters.Params(system=system))
    _assert_ccw_ring_of_mesh_edges(polygons, polygons.boundary_inds.tolist())


def test_get_system_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid initial system"):
        init_systems.get_system(parameters.Params(system="invalid"))


def test_single_polygon() -> None:
    polygons = init_systems.get_system(parameters.Params(system="single"))
    n_vertices = polygons.init_vertices.shape[0]

    # One extended polygon containing every vertex, all on the boundary
    assert polygons.indices.shape == (1, n_vertices + 2)
    np.testing.assert_array_equal(
        polygons.boundary_inds, np.arange(n_vertices)
    )

    polygon_inds = polygons.indices[0, 1:-1]
    polygon = shapely.Polygon(polygons.init_vertices[polygon_inds])
    assert polygons.mesh_area == pytest.approx(polygon.area)


def test_get_jax_polygons_matches_system() -> None:
    params = parameters.Params(system="single")
    polygons = init_systems.get_system(params)
    jax_polygons = init_systems.get_jax_polygons(params)

    assert np.asarray(jax_polygons.init_vertices) == pytest.approx(
        polygons.init_vertices
    )
    np.testing.assert_array_equal(jax_polygons.indices, polygons.indices)
    assert jax_polygons.mesh_area == pytest.approx(polygons.mesh_area)


def test_get_shapely_polygons_areas_sum_to_mesh_area() -> None:
    polygons = init_systems.get_system(parameters.Params(system="few"))
    shapely_polygons = init_systems.get_shapely_polygons(
        polygons.init_vertices, polygons.indices
    )

    assert len(shapely_polygons) == len(polygons.indices)
    assert all(polygon.is_valid for polygon in shapely_polygons)
    total_area = shapely.union_all(shapely_polygons).area
    assert total_area == pytest.approx(polygons.mesh_area, rel=0.05)


@pytest.mark.parametrize("system", ["few", "many", "full", "single"])
def test_vertex_numbers(system: str) -> None:
    polygons = init_systems.get_system(parameters.Params(system=system))
    vertex_numbers = init_systems.VertexNumbers(polygons)

    basal_inds = np.flatnonzero(~polygons.free_mask[:, 1])
    assert vertex_numbers.basal == len(basal_inds)
    assert vertex_numbers.boundary == len(polygons.boundary_inds)
    # Basal vertices are a subset of the boundary vertices
    assert set(basal_inds) <= set(polygons.boundary_inds)
    # The two basal corners are counted as both basal and non-basal
    assert (
        vertex_numbers.non_basal_with_corners
        == vertex_numbers.boundary - vertex_numbers.basal + 2
    )


def test_knots() -> None:
    knots = init_systems.Knots()
    origin_x = init_systems.Coords.base_origin[0]

    assert knots.left_knots.ndim == 2 and knots.left_knots.shape[1] == 2
    assert knots.right_knots.shape == knots.left_knots.shape
    assert knots.all_knots == pytest.approx(
        np.concatenate(
            [knots.left_knots, knots.center_knots, knots.right_knots]
        )
    )

    # Left and right knots mirror each other around the mid axis
    assert np.all(knots.left_knots[:, 0] < origin_x)
    assert knots.right_knots[:, 0] - origin_x == pytest.approx(
        origin_x - knots.left_knots[:, 0]
    )
    assert knots.right_knots[:, 1] == pytest.approx(knots.left_knots[:, 1])

    # One center knot on the mid axis per row of side knots
    assert knots.center_knots[:, 0] == pytest.approx(origin_x)
    assert knots.center_knots[:, 1] == pytest.approx(
        np.unique(knots.left_knots[:, 1])
    )
