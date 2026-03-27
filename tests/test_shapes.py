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


def _assert_no_duplicates_in_target_boundary(params):
    polygons = init_systems.get_system(params)
    vertex_numbers = init_systems.VertexNumbers(polygons)
    target_boundary = shapes.get_target_boundary(
        params, polygons.mesh_area, vertex_numbers
    )

    n_pairs = _count_pairs_of_duplicates(target_boundary.vertices)
    assert n_pairs == 0


def test_target_boundaries_have_no_duplicate_points():
    misc_shapes = ["petal", "long_petal", "nconv"]
    for shape in misc_shapes:
        params = parameters.Params(shape=shape)
        _assert_no_duplicates_in_target_boundary(params)

    trapezoid_angles = [61.0, 75.0, 90.0, 120.0]
    for angle in trapezoid_angles:
        params = parameters.Params(trapezoid_angle=angle)
        _assert_no_duplicates_in_target_boundary(params)
