import numpy as np

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, shapes

from scipy.spatial import KDTree


def _count_pairs_of_duplicates(points, tol=1e-6):
    tree = KDTree(points)
    pairs = tree.query_pairs(r=tol)
    n_duplicates = len(pairs)
    return n_duplicates


def _assert_n_pairs_of_duplicates(points, expected):
    n_duplicates = _count_pairs_of_duplicates(points)
    assert n_duplicates == expected


def test_duplicate_counter():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    _assert_n_pairs_of_duplicates(points, 0)

    points = np.array([[0.0, 0.0], [0.0, 0.0]])
    _assert_n_pairs_of_duplicates(points, 1)

    points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    _assert_n_pairs_of_duplicates(points, 2)

    points = np.array([[0.0, 0.0], [0.0, 0.0 + 1e-8]])
    _assert_n_pairs_of_duplicates(points, 1)


def _assert_no_duplicates_in_target_boundary(shape):
    general_params = parameters.Params(shape=shape)
    polygons = init_systems.get_system(general_params)
    vertex_numbers = init_systems.VertexNumbers(polygons)
    target_boundary = shapes.get_target_boundary(
        shape, polygons.mesh_area, vertex_numbers
    )

    n_pairs = _count_pairs_of_duplicates(target_boundary.vertices)
    assert n_pairs == 0


def test_target_boundaries_have_no_duplicate_points():
    test_shapes = [
        "nconv",
        "petal",
        "long_petal",
        "trapezoid",
        "narrow",
        "square",
        "wide",
        "triangle",
    ]
    for shape in test_shapes:
        _assert_no_duplicates_in_target_boundary(shape)
