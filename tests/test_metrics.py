import numpy as np

from diff_tissue.core import init_systems, metrics
from diff_tissue.app import parameters


def _assert(vertices, indices, expected):
    n_crossings = metrics.count_edge_crossings(vertices, indices)
    assert n_crossings == expected


def _test_touching_triangles():
    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.0], [1.5, 1.0]]
    )
    indices = [[0, 1, 2], [2, 3, 4]]
    _assert(vertices, indices, expected=0)


def _test_non_overlapping_squares():
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [3.0, 1.0],
        ]
    )
    indices = [[0, 1, 2, 3], [4, 5, 6, 7]]
    _assert(vertices, indices, expected=0)


def _test_two_overlapping_squares():
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -1.0],
            [3.5, -1.0],
            [3.5, 2.0],
            [0.5, 2.0],
        ]
    )
    indices = [[0, 1, 2, 3], [4, 5, 6, 7]]
    _assert(vertices, indices, expected=2)


def _test_three_overlapping_squares():
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -1.0],
            [3.5, -1.0],
            [3.5, 2.0],
            [0.5, 2.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [3.0, 1.0],
        ]
    )
    indices = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    _assert(vertices, indices, expected=4)


def _test_init_mesh():
    polygons = init_systems.get_system(parameters.Params())
    valid_inds = init_systems.make_poly_idx_lists(polygons.indices)
    _assert(polygons.init_vertices, valid_inds, expected=0)


def test_count_edge_crossings():
    _test_touching_triangles()
    _test_non_overlapping_squares()
    _test_two_overlapping_squares()
    _test_three_overlapping_squares()
    _test_init_mesh()
