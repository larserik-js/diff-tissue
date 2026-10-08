import numpy as np
import pytest

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, metrics


_UNIT_SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
_WIDE_SQUARE = [[0.5, -1.0], [3.5, -1.0], [3.5, 2.0], [0.5, 2.0]]
_SHIFTED_SQUARE = [[3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0]]


def _assert_n_crossings(vertices, indices, expected):
    n_crossings = metrics.count_edge_crossings(vertices, indices)
    assert n_crossings == expected


@pytest.mark.parametrize(
    ("vertices", "indices", "expected"),
    [
        pytest.param(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.0], [1.5, 1.0]],
            [[0, 1, 2], [2, 3, 4]],
            0,
            id="touching_triangles",
        ),
        pytest.param(
            _UNIT_SQUARE + _SHIFTED_SQUARE,
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            0,
            id="non_overlapping_squares",
        ),
        pytest.param(
            _UNIT_SQUARE + _WIDE_SQUARE,
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            2,
            id="two_overlapping_squares",
        ),
        pytest.param(
            _UNIT_SQUARE + _WIDE_SQUARE + _SHIFTED_SQUARE,
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            4,
            id="three_overlapping_squares",
        ),
    ],
)
def test_count_edge_crossings(vertices, indices, expected):
    _assert_n_crossings(np.array(vertices), indices, expected)


def test_count_edge_crossings_init_mesh():
    polygons = init_systems.get_system(parameters.Params())
    valid_inds = init_systems.make_poly_idx_lists(polygons.indices)
    _assert_n_crossings(polygons.init_vertices, valid_inds, expected=0)


def _make_square_and_triangle():
    vertices = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    # Extended, counter-clockwise polygons, padded with -1
    indices = np.array([[3, 0, 1, 2, 3, 0], [6, 4, 5, 6, 4, -1]])
    valid_mask = indices != -1
    return vertices, indices, valid_mask


def test_initialize_poly_metrics():
    vertices, indices, valid_mask = _make_square_and_triangle()
    poly_metrics = metrics.initialize_poly_metrics(
        vertices, indices, valid_mask
    )

    np.testing.assert_allclose(poly_metrics.areas, [2.0, 0.5])
    # Rectangle is wider than tall: x var 1.0, y var 0.25
    np.testing.assert_allclose(
        poly_metrics.anisotropies[0], (0.25 - 1.0) / 1.25, rtol=1e-6
    )
    np.testing.assert_allclose(
        poly_metrics.optimal_angles[:, 0], [np.pi / 2, 2 * np.pi / 3]
    )
    # Right angles between all consecutive rectangle edges
    np.testing.assert_allclose(poly_metrics.masked_cosines[0], 0.0, atol=1e-5)


def test_update_poly_metrics_scales_areas():
    vertices, indices, valid_mask = _make_square_and_triangle()
    poly_metrics = metrics.initialize_poly_metrics(
        vertices, indices, valid_mask
    )
    updated = metrics.update_poly_metrics(poly_metrics, 2.0 * vertices)

    np.testing.assert_allclose(updated.areas, 4.0 * poly_metrics.areas)
    np.testing.assert_allclose(
        updated.anisotropies, poly_metrics.anisotropies, rtol=1e-6
    )


def test_get_tutte_metrics():
    params = parameters.Params()
    tutte_metrics = metrics.get_tutte_metrics(params)
    polygons = init_systems.get_system(params)
    n_polygons = len(polygons.indices)

    assert tutte_metrics.vertices.shape == polygons.init_vertices.shape
    assert tutte_metrics.centroids.shape == (n_polygons, 2)
    assert tutte_metrics.anisotropies.shape == (n_polygons,)
    assert np.all(np.abs(tutte_metrics.anisotropies) <= 1.0)
    # Tutte mapping fills the target shape, which has the mesh area
    assert np.all(tutte_metrics.areas > 0.0)
    assert float(np.sum(tutte_metrics.areas)) == pytest.approx(
        polygons.mesh_area, rel=0.05
    )
