import numpy as np
import pytest
from scipy.spatial import KDTree

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, shapes


def _count_pairs_of_duplicates(points, tol=1e-6):
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
def test_duplicate_counter(points, expected):
    assert _count_pairs_of_duplicates(np.array(points)) == expected


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(parameters.Params(shape=shape), id=shape)
        for shape in ["petal", "long_petal", "nconv"]
    ]
    + [
        pytest.param(
            parameters.Params(trapezoid_angle=angle), id=f"trapezoid_{angle}"
        )
        for angle in [61.0, 75.0, 90.0, 120.0]
    ],
)
def test_target_boundary_has_no_duplicate_points(params):
    polygons = init_systems.get_system(params)
    target_boundary = shapes.get_target_boundary(params, polygons)
    assert _count_pairs_of_duplicates(target_boundary.vertices) == 0
