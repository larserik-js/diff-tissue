import jax
import jax.numpy as jnp
from jaxopt import LBFGS

from . import my_utils


def _calc_areas_loss(target_areas, areas, proximal_mask):
    areas = jnp.where(proximal_mask, 0.66 * areas, areas)
    areas_loss = jnp.sum((target_areas - areas)**2)
    return areas_loss


def _calc_all_angles_loss(all_cells, valid_mask, optimal_angles):
    edges = all_cells[:, 1:] - all_cells[:, :-1]
    epsilon = 1e-7
    norms = jnp.linalg.norm(edges + epsilon, axis=2)
    dot_products = jnp.sum(edges[:, :-1] * edges[:, 1:], axis=2)

    cosines = dot_products / (epsilon + norms[:, :-1] * norms[:, 1:])
    clip_value = 1.0 - epsilon
    cosines = jnp.clip(cosines, -clip_value, clip_value)

    optimal_cosines = jnp.cos(optimal_angles)
    valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    valid = valid[:, 1:] & valid[:, :-1]

    angles_loss = jnp.sum((cosines - optimal_cosines)**2 * valid)

    return angles_loss


def _calc_elongations_loss(target_elongations, elongations, proximal_mask):
    elongations = jnp.where(
        proximal_mask, 0.66 * elongations, elongations
    )
    elongation_diffs = target_elongations - elongations
    elongations_loss = jnp.sum(jnp.square(elongation_diffs))
    return elongations_loss


def _calc_growth_loss(vertices, target_areas, target_elongations,
                      optimal_angles, jax_arrays, params):
    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])

    areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])
    elongations = my_utils.calc_elongations(all_cells, jax_arrays['valid_mask'])

    areas_loss = params['areas_loss_weight'] * _calc_areas_loss(
        target_areas, areas, jax_arrays['proximal_mask']
    )
    angles_loss = params['angles_loss_weight'] * _calc_all_angles_loss(
        all_cells, jax_arrays['valid_mask'], optimal_angles
    )
    elongations_loss = (
        params['elongation_loss_weight'] * _calc_elongations_loss(
            target_elongations, elongations, jax_arrays['proximal_mask']
        )
    )

    loss = areas_loss + angles_loss + elongations_loss

    return loss


def _update_targets(init_targets, goals, t_frac):
    targets = init_targets + (goals - init_targets) * t_frac
    return targets


def _lbfgs_solve(vertices, target_areas, target_elongations, optimal_angles,
                 jax_arrays, params):
    solver = LBFGS(fun=_calc_growth_loss, maxiter=50)
    result = solver.run(
        vertices, target_areas, target_elongations, optimal_angles, jax_arrays,
        params
    )
    updated_vertices = result.params

    return updated_vertices


def _update_vertices(vertices, t, init_areas, goal_areas, init_elongations,
                     goal_elongations, optimal_angles, jax_arrays, params):
    t_frac = t / params['n_growth_steps']
    target_areas = _update_targets(init_areas, goal_areas, t_frac)
    target_elongations = _update_targets(
        init_elongations, goal_elongations, t_frac
    )

    updated_vertices = _lbfgs_solve(
        vertices, target_areas, target_elongations, optimal_angles, jax_arrays,
        params
    )
    updated_vertices = jnp.where(
        jax_arrays['free_mask'], updated_vertices, jax_arrays['init_vertices']
    )

    return updated_vertices


def iterate(goal_areas, goal_elongations, n_steps, jax_arrays, params):
    init_vertices = jax_arrays['init_vertices']

    all_cells = my_utils.get_all_cells(init_vertices, jax_arrays['indices'])
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])
    init_elongations = my_utils.calc_elongations(
        all_cells, jax_arrays['valid_mask']
    )
    optimal_angles = my_utils.calc_optimal_angles(jax_arrays['valid_mask'])

    def update_step(carry, t):
        (vertices, init_areas, init_elongations, goal_areas,
         goal_elongations) = carry

        vertices = _update_vertices(
            vertices, t, init_areas, goal_areas, init_elongations,
            goal_elongations, optimal_angles, jax_arrays, params
        )

        carry = (
            vertices, init_areas, init_elongations, goal_areas, goal_elongations
        )

        return carry, vertices

    init_carry = (
        init_vertices, init_areas, init_elongations, goal_areas,
        goal_elongations
    )
    _, growth_evolution = jax.lax.scan(
        update_step, init_carry, jnp.arange(n_steps)
    )

    return growth_evolution
