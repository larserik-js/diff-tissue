from .jax_bootstrap import jax, jaxopt, jnp
from . import my_utils


def _calc_areas_loss(target_areas, areas):
    areas_loss = jnp.sum((target_areas - areas)**2)
    return areas_loss


def _calc_angles_loss(masked_cosines, optimal_angles):
    optimal_cosines = jnp.cos(optimal_angles)
    squared_diffs = jnp.square(masked_cosines - optimal_cosines)
    angles_loss = jnp.nansum(squared_diffs)
    return angles_loss


def _calc_anisotropies_loss(target_anisotropies, anisotropies):
    anisotropy_diffs = target_anisotropies - anisotropies
    anisotropies_loss = jnp.sum(jnp.square(anisotropy_diffs))
    return anisotropies_loss


def _calc_growth_loss(
        vertices, target_areas, target_anisotropies, optimal_angles,
        poly_metrics, params
    ):
    poly_metrics = poly_metrics.update(vertices)

    areas_loss = params.areas_loss_weight * _calc_areas_loss(
        target_areas, poly_metrics.areas
    )
    angles_loss = params.angles_loss_weight * _calc_angles_loss(
        poly_metrics.masked_cosines, optimal_angles
    )
    anisotropies_loss = (
        params.anisotropy_loss_weight * _calc_anisotropies_loss(
            target_anisotropies, poly_metrics.anisotropies
        )
    )

    loss = areas_loss + angles_loss + anisotropies_loss

    return loss


def _update_targets(init_targets, goals, t_frac):
    targets = init_targets + (goals - init_targets) * t_frac
    return targets


def _lbfgs_solve(
        vertices, target_areas, target_anisotropies, optimal_angles,
        poly_metrics, params
    ):
    solver = jaxopt.LBFGS(fun=_calc_growth_loss, maxiter=50)
    result = solver.run(
        vertices, target_areas, target_anisotropies, optimal_angles,
        poly_metrics, params
    )
    updated_vertices = result.params

    return updated_vertices


def _update_vertices(
        vertices, t, goal_areas, goal_anisotropies, init_areas,
        init_anisotropies, optimal_angles, poly_metrics, jax_arrays, params
    ):
    t_frac = t / params.n_growth_steps
    target_areas = _update_targets(init_areas, goal_areas, t_frac)
    target_anisotropies = _update_targets(
        init_anisotropies, goal_anisotropies, t_frac
    )

    updated_vertices = _lbfgs_solve(
        vertices, target_areas, target_anisotropies, optimal_angles,
        poly_metrics, params
    )
    updated_vertices = jnp.where(
        jax_arrays['free_mask'], updated_vertices, jax_arrays['init_vertices']
    )

    return updated_vertices


def iterate(goal_areas, goal_anisotropies, n_steps, jax_arrays, params):
    init_vertices = jax_arrays['init_vertices']

    init_areas = jax_arrays['init_areas']
    init_anisotropies = jax_arrays['init_anisotropies']
    poly_metrics = my_utils.PolyMetrics.create(
        init_vertices, jax_arrays['indices'], jax_arrays['valid_mask']
    )
    optimal_angles = my_utils.calc_optimal_angles(jax_arrays['valid_mask'])

    def update_step(carry, t):
        vertices, poly_metrics, goal_areas, goal_anisotropies = carry

        vertices = _update_vertices(
            vertices, t, goal_areas, goal_anisotropies, init_areas, 
            init_anisotropies, optimal_angles, poly_metrics, jax_arrays, params
        )

        carry = (vertices, poly_metrics, goal_areas, goal_anisotropies)

        return carry, vertices

    init_carry = (init_vertices, poly_metrics, goal_areas, goal_anisotropies)

    _, growth_evolution = jax.lax.scan(
        update_step, init_carry, jnp.arange(n_steps)
    )

    return growth_evolution
