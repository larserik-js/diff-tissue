from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import shapely

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, shapes, tutte_fields


_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [
        pytest.param(_UNIT_SQUARE, 9, id="square"),
        # Includes points on the hypotenuse
        pytest.param(_TRIANGLE, 6, id="triangle"),
    ],
)
def test_get_points_inside_shape(
    boundary: npt.NDArray[np.floating], expected: int
) -> None:
    points = tutte_fields.get_points_inside_shape(boundary, nx=3, ny=3)

    assert points.shape == (expected, 2)
    shape = shapely.Polygon(boundary)
    assert np.all(shapely.covers(shape, shapely.points(points)))


def _make_mesh(
    areas: tuple[float, float], anisotropies: tuple[float, float]
) -> tutte_fields.Mesh:
    """A unit square split into a left and a right half."""
    left = shapely.box(0.0, 0.0, 0.5, 1.0)
    right = shapely.box(0.5, 0.0, 1.0, 1.0)
    return tutte_fields.Mesh(
        polygons=[left, right],
        areas=np.array(areas),
        anisotropies=np.array(anisotropies),
    )


_TWO_MESHES = [
    _make_mesh(areas=(1.0, 2.0), anisotropies=(0.1, -0.2)),
    _make_mesh(areas=(3.0, 4.0), anisotropies=(0.3, 0.4)),
]
# One point in the left half and one in the right half
_INSIDE_POINTS = np.array([[0.25, 0.5], [0.75, 0.5]])


def test_get_fields_averages_areas() -> None:
    mean_areas, _ = tutte_fields.get_fields(_TWO_MESHES, _INSIDE_POINTS)
    np.testing.assert_allclose(mean_areas, [2.0, 3.0])


def test_get_fields_averages_anisotropies() -> None:
    _, mean_anisotropies = tutte_fields.get_fields(_TWO_MESHES, _INSIDE_POINTS)
    np.testing.assert_allclose(mean_anisotropies, [0.2, 0.1])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_get_fields_leaves_points_outside_meshes_unassigned() -> None:
    outside_points = np.array([[5.0, 5.0]])
    mean_areas, mean_anisotropies = tutte_fields.get_fields(
        _TWO_MESHES, outside_points
    )
    assert np.all(np.isnan(mean_areas))
    assert np.all(np.isnan(mean_anisotropies))


_N_MESHES = 2


@dataclass
class _BuiltMeshes:
    meshes: list[tutte_fields.Mesh]
    # Target shape for the mesh with the same index (seed)
    targets: list[shapely.Polygon]


@pytest.fixture(scope="module")
def built_meshes() -> _BuiltMeshes:
    params = parameters.Params()
    meshes = tutte_fields.build_meshes(params, _N_MESHES)

    targets = []
    for seed in range(_N_MESHES):
        seed_params = params.replace(seed=seed)
        polygons = init_systems.get_system(seed_params)
        target = shapes.get_target_boundary(seed_params, polygons)
        targets.append(shapely.Polygon(target.vertices))

    return _BuiltMeshes(meshes, targets)


def test_build_meshes_builds_one_mesh_per_seed(
    built_meshes: _BuiltMeshes,
) -> None:
    assert len(built_meshes.meshes) == _N_MESHES


def test_build_meshes_areas_match_polygons(
    built_meshes: _BuiltMeshes,
) -> None:
    for mesh in built_meshes.meshes:
        polygon_areas = [polygon.area for polygon in mesh.polygons]
        np.testing.assert_allclose(mesh.areas, polygon_areas, rtol=1e-9)


def test_build_meshes_polygons_cover_target_shape(
    built_meshes: _BuiltMeshes,
) -> None:
    for mesh, target in zip(built_meshes.meshes, built_meshes.targets):
        union = shapely.union_all(mesh.polygons)
        uncovered_area = union.symmetric_difference(target).area
        assert uncovered_area == pytest.approx(0.0, abs=1e-9 * target.area)


def test_build_meshes_anisotropies_are_in_range(
    built_meshes: _BuiltMeshes,
) -> None:
    for mesh in built_meshes.meshes:
        assert np.all(np.abs(mesh.anisotropies) <= 1.0)


def test_fields_cover_whole_target_shape(built_meshes: _BuiltMeshes) -> None:
    target_boundary = np.asarray(built_meshes.targets[0].exterior.coords)
    points = tutte_fields.get_points_inside_shape(
        target_boundary, nx=30, ny=30
    )

    mean_areas, mean_anisotropies = tutte_fields.get_fields(
        built_meshes.meshes, points
    )

    assert not np.any(np.isnan(mean_areas))
    assert not np.any(np.isnan(mean_anisotropies))
