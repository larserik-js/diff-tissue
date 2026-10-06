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
