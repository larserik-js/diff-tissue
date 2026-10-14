from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, metrics, morphing


def test_update_targets_interpolates() -> None:
    init, goals = np.array([0.0, 2.0]), np.array([1.0, 4.0])
    np.testing.assert_allclose(
        morphing._update_targets(init, goals, 0.5), [0.5, 3.0]
    )


@dataclass
class _StepTargets:
    areas: npt.NDArray[np.floating]
    anisotropies: npt.NDArray[np.floating]


_INIT_AREAS = np.array([1.0, 2.0])
_GOAL_AREAS = np.array([3.0, 1.0])
_INIT_ANISOTROPIES = np.array([0.0, 0.5])
_GOAL_ANISOTROPIES = np.array([-0.5, 0.25])
_N_UNIT_STEPS = 4


def _get_step_targets(t: int, monkeypatch: pytest.MonkeyPatch) -> _StepTargets:
    """Returns the targets that morph step t passes to the solver."""
    recorded = []

    def recording_solve(
        vertices: Any,
        target_areas: Any,
        target_anisotropies: Any,
        *_: Any,
    ) -> Any:
        recorded.append(_StepTargets(target_areas, target_anisotropies))
        return vertices

    monkeypatch.setattr(morphing, "_lbfgs_solve", recording_solve)
    vertices = np.zeros((3, 2))
    polygons = SimpleNamespace(
        free_mask=np.ones_like(vertices, dtype=bool), init_vertices=vertices
    )
    morphing._update_vertices(
        vertices,
        t,
        _N_UNIT_STEPS,
        _GOAL_AREAS,
        _GOAL_ANISOTROPIES,
        _INIT_AREAS,
        _INIT_ANISOTROPIES,
        None,
        None,
        polygons,
    )
    return recorded[0]


def test_update_vertices_first_step_moves_targets_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _get_step_targets(0, monkeypatch)
    step_frac = 1 / _N_UNIT_STEPS
    np.testing.assert_allclose(
        targets.areas, _INIT_AREAS + step_frac * (_GOAL_AREAS - _INIT_AREAS)
    )
    np.testing.assert_allclose(
        targets.anisotropies,
        _INIT_ANISOTROPIES
        + step_frac * (_GOAL_ANISOTROPIES - _INIT_ANISOTROPIES),
    )


def test_update_vertices_last_step_targets_goals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _get_step_targets(_N_UNIT_STEPS - 1, monkeypatch)
    np.testing.assert_allclose(targets.areas, _GOAL_AREAS)
    np.testing.assert_allclose(targets.anisotropies, _GOAL_ANISOTROPIES)


# The solver does not converge within a step, so the outcome is sensitive to
# platform-level rounding. Small steps keep it reliably moving to the goals
_N_STEPS = 50


@dataclass
class _MorphRun:
    init_vertices: npt.NDArray[np.floating]
    morph_evolution: npt.NDArray[np.floating]
    fixed_mask: npt.NDArray[np.bool_]
    init_areas: npt.NDArray[np.floating]
    goal_areas: npt.NDArray[np.floating]
    final_areas: npt.NDArray[np.floating]
    init_anisotropies: npt.NDArray[np.floating]
    goal_anisotropies: npt.NDArray[np.floating]
    final_anisotropies: npt.NDArray[np.floating]

    @property
    def final_vertices(self) -> npt.NDArray[np.floating]:
        return self.morph_evolution[-1]


@pytest.fixture(
    scope="module",
    params=[
        # (area scale factor, goal anisotropy)
        pytest.param((0.5, -0.5), id="shrink_wider"),
        pytest.param((1.5, 0.5), id="grow_taller"),
    ],
)
def morph_run(request: pytest.FixtureRequest) -> _MorphRun:
    """Morphs the default system towards scaled areas and anisotropies."""
    area_factor, goal_anisotropy = request.param
    params = parameters.Params(n_morph_steps=_N_STEPS)
    polygons = init_systems.get_jax_polygons(params)
    poly_metrics = metrics.initialize_poly_metrics(
        polygons.init_vertices, polygons.indices, polygons.valid_mask
    )
    goal_areas = area_factor * poly_metrics.areas
    goal_anisotropies = np.full_like(
        poly_metrics.anisotropies, goal_anisotropy
    )

    morph_evolution = np.asarray(
        morphing.iterate(
            goal_areas, goal_anisotropies, _N_STEPS, polygons, params
        )
    )
    final_metrics = metrics.update_poly_metrics(
        poly_metrics, morph_evolution[-1]
    )

    return _MorphRun(
        init_vertices=np.asarray(polygons.init_vertices),
        morph_evolution=morph_evolution,
        fixed_mask=~np.asarray(polygons.free_mask),
        init_areas=np.asarray(poly_metrics.areas),
        goal_areas=np.asarray(goal_areas),
        final_areas=np.asarray(final_metrics.areas),
        init_anisotropies=np.asarray(poly_metrics.anisotropies),
        goal_anisotropies=np.asarray(goal_anisotropies),
        final_anisotropies=np.asarray(final_metrics.anisotropies),
    )


@pytest.mark.integration
def test_iterate_returns_one_state_per_step(morph_run: _MorphRun) -> None:
    expected_shape = (_N_STEPS, *morph_run.init_vertices.shape)
    assert morph_run.morph_evolution.shape == expected_shape


@pytest.mark.integration
def test_iterate_vertices_are_finite(morph_run: _MorphRun) -> None:
    assert np.all(np.isfinite(morph_run.morph_evolution))


@pytest.mark.integration
def test_iterate_keeps_fixed_vertices(morph_run: _MorphRun) -> None:
    fixed_mask = morph_run.fixed_mask
    np.testing.assert_allclose(
        morph_run.final_vertices[fixed_mask],
        morph_run.init_vertices[fixed_mask],
    )


@pytest.mark.integration
def test_iterate_moves_areas_towards_goal_areas(morph_run: _MorphRun) -> None:
    init_dist = np.linalg.norm(morph_run.init_areas - morph_run.goal_areas)
    final_dist = np.linalg.norm(morph_run.final_areas - morph_run.goal_areas)
    assert final_dist < init_dist


@pytest.mark.integration
def test_iterate_moves_anisotropies_towards_goal_anisotropies(
    morph_run: _MorphRun,
) -> None:
    init_dist = np.linalg.norm(
        morph_run.init_anisotropies - morph_run.goal_anisotropies
    )
    final_dist = np.linalg.norm(
        morph_run.final_anisotropies - morph_run.goal_anisotropies
    )
    assert final_dist < init_dist
