from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, metrics, shape_opt


_N_VERTICES = 6
_N_POLYGONS = 3


def _make_sim_states(
    loss_vals: list[float], valid: list[bool], n_edge_crossings: list[int]
) -> shape_opt.SimStates:
    """Sim states with the given losses and validity, and random values for
    every other field."""
    n_states = len(loss_vals)
    rng = np.random.default_rng(seed=0)

    def random_values(*shape: int) -> npt.NDArray[np.floating]:
        return rng.random((n_states, *shape))

    return shape_opt.SimStates(
        loss_vals=np.array(loss_vals),
        shape_loss_vals=random_values(),
        var_loss_vals=random_values(),
        poly_id_loss_vals=random_values(),
        valid=np.array(valid),
        final_vertices=random_values(_N_VERTICES, 2),
        goal_areas=random_values(_N_POLYGONS),
        goal_anisotropies=random_values(_N_POLYGONS),
        final_areas=random_values(_N_POLYGONS),
        final_anisotropies=random_values(_N_POLYGONS),
        n_edge_crossings=np.array(n_edge_crossings),
    )


@pytest.mark.parametrize(
    ("loss_vals", "valid", "n_edge_crossings", "expected"),
    [
        pytest.param([1.0], [True], [0], 0, id="single_valid"),
        pytest.param([1.0], [False], [0], 0, id="single_invalid"),
        pytest.param([0.5, 0.1], [False, False], [0, 0], 1, id="two_invalid"),
        pytest.param([1.0, 1.0], [True, False], [0, 0], 0, id="mixed_valid"),
        pytest.param(
            [np.inf, np.inf],
            [False, False],
            [5, 5],
            0,
            id="all_inf_losses",
        ),
    ],
)
def test_get_valid_best_idx(
    loss_vals: list[float],
    valid: list[bool],
    n_edge_crossings: list[int],
    expected: int,
) -> None:
    sim_states = _make_sim_states(loss_vals, valid, n_edge_crossings)
    assert shape_opt._get_valid_best_idx(sim_states) == expected


def test_get_best_state_takes_all_fields_from_best_valid_state() -> None:
    # The lowest loss is invalid, so the best valid state is the last one
    sim_states = _make_sim_states(
        [0.5, 0.1, 0.3], [True, False, True], [0, 1, 0]
    )
    best_idx = 2

    best = shape_opt.get_best_state(sim_states)

    assert best.loss == pytest.approx(sim_states.loss_vals[best_idx])
    assert best.shape_loss == pytest.approx(
        sim_states.shape_loss_vals[best_idx]
    )
    assert best.var_loss == pytest.approx(sim_states.var_loss_vals[best_idx])
    assert best.poly_id_loss == pytest.approx(
        sim_states.poly_id_loss_vals[best_idx]
    )
    np.testing.assert_allclose(
        best.final_vertices, sim_states.final_vertices[best_idx]
    )
    np.testing.assert_allclose(
        best.goal_areas, sim_states.goal_areas[best_idx]
    )
    np.testing.assert_allclose(
        best.goal_anisotropies, sim_states.goal_anisotropies[best_idx]
    )
    np.testing.assert_allclose(
        best.final_areas, sim_states.final_areas[best_idx]
    )
    np.testing.assert_allclose(
        best.final_anisotropies, sim_states.final_anisotropies[best_idx]
    )


def test_sigmoid_inverse_roundtrip() -> None:
    vals = np.array([0.1, 2.5, 4.9])
    logits = shape_opt._calc_inverse_sigmoid(0.0, 5.0, vals)
    np.testing.assert_allclose(shape_opt._calc_sigmoid(0.0, 5.0, logits), vals)


@pytest.mark.parametrize(
    ("areas", "n_edge_crossings", "expected"),
    [
        pytest.param([1.0, 2.0], 0, True, id="valid"),
        pytest.param([1.0, -2.0], 0, False, id="negative_area"),
        pytest.param([1.0, 2.0], 1, False, id="edge_crossing"),
    ],
)
def test_validate(
    areas: list[float], n_edge_crossings: int, expected: bool
) -> None:
    assert shape_opt._validate(np.array(areas), n_edge_crossings) == expected


@dataclass
class _InitCtx:
    tutte_metrics: metrics.TutteMetrics
    goal_area_bounds: tuple[float, float]
    knots: init_systems.Knots
    logits: shape_opt._Logits

    @property
    def clipped_tutte_areas(self) -> npt.NDArray[np.floating]:
        min_area, max_area = self.goal_area_bounds
        areas = np.asarray(self.tutte_metrics.areas)
        return np.clip(areas, min_area + 1e-8, max_area - 1e-8)


def _make_init_ctx(use_knots: bool) -> _InitCtx:
    params = parameters.Params(knots=use_knots)
    polygons = init_systems.get_system(params)
    tutte_metrics = metrics.get_tutte_metrics(params)
    goal_area_bounds = shape_opt._calc_goal_area_bounds(polygons)
    knots = init_systems.Knots()
    logits = shape_opt._get_init_logits(
        goal_area_bounds, knots, tutte_metrics, params
    )
    return _InitCtx(tutte_metrics, goal_area_bounds, knots, logits)


@pytest.fixture(scope="module")
def poly_init() -> _InitCtx:
    return _make_init_ctx(use_knots=False)


@pytest.fixture(scope="module")
def knot_init() -> _InitCtx:
    return _make_init_ctx(use_knots=True)


def test_poly_init_logits_give_tutte_areas(poly_init: _InitCtx) -> None:
    goal_areas = shape_opt._calc_goal_areas(
        poly_init.goal_area_bounds, poly_init.logits.ar_logits, False, None
    )
    np.testing.assert_allclose(
        goal_areas, poly_init.clipped_tutte_areas, rtol=1e-6
    )


def test_poly_init_logits_give_tutte_anisotropies(
    poly_init: _InitCtx,
) -> None:
    goal_anisotropies = shape_opt._calc_goal_anisotropies(
        poly_init.logits.an_logits, False, None
    )
    np.testing.assert_allclose(
        goal_anisotropies, poly_init.tutte_metrics.anisotropies, atol=1e-6
    )


def _closest_polygon_inds(
    knot_init: _InitCtx,
) -> npt.NDArray[np.integer]:
    """Polygons closest to the left knots, then the center knots."""
    knots = np.concatenate(
        [knot_init.knots.left_knots, knot_init.knots.center_knots]
    )
    centroids = np.asarray(knot_init.tutte_metrics.centroids)
    dists = np.linalg.norm(centroids[None, :] - knots[:, None], axis=2)
    return np.argmin(dists, axis=1)


def test_knot_init_logits_give_closest_polygon_areas(
    knot_init: _InitCtx,
) -> None:
    closest_inds = _closest_polygon_inds(knot_init)
    goal_areas = shape_opt._calc_goal_areas(
        knot_init.goal_area_bounds, knot_init.logits.ar_logits, False, None
    )
    np.testing.assert_allclose(
        goal_areas, knot_init.clipped_tutte_areas[closest_inds], rtol=1e-6
    )


def test_knot_init_logits_give_closest_polygon_anisotropies(
    knot_init: _InitCtx,
) -> None:
    closest_inds = _closest_polygon_inds(knot_init)
    goal_anisotropies = shape_opt._calc_goal_anisotropies(
        knot_init.logits.an_logits, False, None
    )
    tutte_anisotropies = np.asarray(knot_init.tutte_metrics.anisotropies)
    np.testing.assert_allclose(
        goal_anisotropies, tutte_anisotropies[closest_inds], atol=1e-6
    )


def test_knot_init_smoothing_stds_are_half_the_knot_spacing(
    knot_init: _InitCtx,
) -> None:
    knots = knot_init.knots
    x_spacing = abs(knots.center_knots[0, 0] - knots.left_knots[0, 0])
    y_spacing = np.diff(np.unique(knots.left_knots[:, 1]))[0]

    assert knot_init.logits.std_logits is not None
    smoothing_stds = shape_opt._calc_smoothing_stds(
        knot_init.logits.std_logits
    )
    np.testing.assert_allclose(
        smoothing_stds, [x_spacing / 2, y_spacing / 2], rtol=1e-6
    )


def test_get_knot_ctx_without_knots_is_none(knot_init: _InitCtx) -> None:
    knot_ctx = shape_opt._get_knot_ctx(
        False, knot_init.knots, knot_init.tutte_metrics.centroids
    )
    assert knot_ctx is None


_N_SHAPE_STEPS = 3


@dataclass
class _ShapeOptRun:
    sim_states: shape_opt.SimStates
    n_vertices: int
    n_polygons: int


@pytest.fixture(scope="module")
def shape_opt_run() -> _ShapeOptRun:
    """Short optimization with knots and mid/outer polygon identities."""
    params = parameters.Params(
        knots=True,
        poly_id_cfg=2,
        n_shape_steps=_N_SHAPE_STEPS,
        n_morph_steps=2,
    )
    polygons = init_systems.get_system(params)
    return _ShapeOptRun(
        sim_states=shape_opt.run(params),
        n_vertices=polygons.init_vertices.shape[0],
        n_polygons=len(polygons.indices),
    )


@pytest.mark.integration
def test_run_stores_one_state_per_step(shape_opt_run: _ShapeOptRun) -> None:
    sim_states = shape_opt_run.sim_states
    n_polygons = shape_opt_run.n_polygons

    for name in ["loss_vals", "valid", "n_edge_crossings"]:
        assert getattr(sim_states, name).shape == (_N_SHAPE_STEPS,), name
    for name in [
        "goal_areas",
        "goal_anisotropies",
        "final_areas",
        "final_anisotropies",
    ]:
        expected_shape = (_N_SHAPE_STEPS, n_polygons)
        assert getattr(sim_states, name).shape == expected_shape, name
    assert sim_states.final_vertices.shape == (
        _N_SHAPE_STEPS,
        shape_opt_run.n_vertices,
        2,
    )


@pytest.mark.integration
def test_run_losses_are_finite(shape_opt_run: _ShapeOptRun) -> None:
    assert np.all(np.isfinite(shape_opt_run.sim_states.loss_vals))


@pytest.mark.integration
def test_run_total_loss_is_sum_of_terms(shape_opt_run: _ShapeOptRun) -> None:
    sim_states = shape_opt_run.sim_states
    np.testing.assert_allclose(
        sim_states.loss_vals,
        sim_states.shape_loss_vals
        + sim_states.var_loss_vals
        + sim_states.poly_id_loss_vals,
    )


@pytest.mark.integration
def test_run_valid_flags_match_areas_and_crossings(
    shape_opt_run: _ShapeOptRun,
) -> None:
    sim_states = shape_opt_run.sim_states
    expected_valid = np.all(sim_states.final_areas >= 0.0, axis=1) & (
        sim_states.n_edge_crossings == 0
    )
    np.testing.assert_array_equal(sim_states.valid, expected_valid)
