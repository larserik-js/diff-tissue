import functools

import jax
import jax.numpy as jnp
from jaxopt import LBFGS
import numpy as np

import my_files, my_utils


def _calc_all_angles_loss(edges, valid_mask, optimal_angles):
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


def _calc_aspect_ratios_loss(target_aspect_ratios, aspect_ratios, basal_mask):
    aspect_ratio_diffs = target_aspect_ratios - aspect_ratios
    aspect_ratios_loss = jnp.sum(jnp.square(basal_mask * aspect_ratio_diffs))
    return aspect_ratios_loss


def _calc_growth_loss(vertices, target_areas, target_aspect_ratios,
                      optimal_angles, jax_arrays, params):
    all_cells = vertices[jax_arrays['indices']]
    edges = all_cells[:, 1:] - all_cells[:, :-1]

    areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])
    aspect_ratios = my_utils.calc_aspect_ratios(
        all_cells, jax_arrays['valid_mask']
    )

    areas_loss = params['areas_loss_weight'] * jnp.sum(
        (target_areas - areas)**2
    )
    angles_loss = params['angles_loss_weight'] * _calc_all_angles_loss(
        edges, jax_arrays['valid_mask'], optimal_angles
    )
    aspect_ratios_loss = (
        params['aspect_ratio_loss_weight'] * _calc_aspect_ratios_loss(
            target_aspect_ratios, aspect_ratios, jax_arrays['basal_mask']
        )
    )

    loss = areas_loss + angles_loss + aspect_ratios_loss

    return loss


_grad_loss = jax.grad(_calc_growth_loss)
_hess_loss = jax.hessian(_calc_growth_loss)


def _update_targets(init_targets, goals, t_frac):
    targets = (
        init_targets + (goals - init_targets) * jnp.sin(0.5 * jnp.pi * t_frac)
    )
    return targets


def _calc_newton_delta(vertices, target_areas, target_aspect_ratios,
                       optimal_angles, jax_arrays, params):
    # (n_vertices, 2)
    grads = _grad_loss(
        vertices, target_areas, target_aspect_ratios, optimal_angles,
        jax_arrays, params
    )
    # (n_vertices, 2, n_vertices, 2)
    H = _hess_loss(
        vertices, target_areas, target_aspect_ratios, optimal_angles,
        jax_arrays, params
    )

    # Symmetrize
    H = 0.5 * (H + jnp.transpose(H, (2,3,0,1)))
    lam = 1e-6
    # Reshape Hessian to (492,492) and grad to (492,)
    H_mat = H.reshape(-1, H.shape[0] * H.shape[1])
    g_vec = grads.reshape(-1)
    delta_vec = jnp.linalg.solve(H_mat + lam * jnp.eye(g_vec.shape[0]), g_vec)

    # Reshape back to (246,2)
    delta = delta_vec.reshape(vertices.shape)
    return delta


def _lbfgs_solve(vertices, target_areas, target_aspect_ratios, optimal_angles,
                jax_arrays, params):
    loss_f = functools.partial(
        _calc_growth_loss,
        target_areas=target_areas,
        target_aspect_ratios=target_aspect_ratios,
        optimal_angles=optimal_angles,
        jax_arrays=jax_arrays,
        params=params
    )
    solver = LBFGS(fun=loss_f, maxiter=200)
    result = solver.run(init_params=vertices)
    updated_vertices = result.params

    return updated_vertices


def _update_vertices(vertices, t, init_areas, goal_areas, init_aspect_ratios,
                     goal_aspect_ratios, optimal_angles, n_steps, jax_arrays,
                     params):
    t_frac = t / n_steps
    target_areas = _update_targets(init_areas, goal_areas, t_frac)
    target_aspect_ratios = _update_targets(
        init_aspect_ratios, goal_aspect_ratios, t_frac
    )

    updated_vertices = _lbfgs_solve(
        vertices, target_areas, target_aspect_ratios, optimal_angles,
        jax_arrays, params
    )
    updated_vertices = jnp.where(
        jax_arrays['free_mask'], updated_vertices, jax_arrays['init_vertices']
    )

    return updated_vertices


@functools.partial(jax.jit, static_argnames=['n_steps'])
def iterate(goal_areas, goal_aspect_ratios, areas, aspect_ratios,
            optimal_angles, n_steps, jax_arrays, params):
    def update_step(carry, t):
        (vertices, areas, aspect_ratios, goal_areas,
         goal_aspect_ratios) = carry

        vertices = _update_vertices(
            vertices, t, areas, goal_areas, aspect_ratios, goal_aspect_ratios,
            optimal_angles, n_steps, jax_arrays, params
        )

        carry = (
            vertices, areas, aspect_ratios, goal_areas, goal_aspect_ratios
        )

        return carry, vertices

    init_carry = (
        jax_arrays['init_vertices'], areas, aspect_ratios, goal_areas,
        goal_aspect_ratios
    )
    _, growth_evolution = jax.lax.scan(
        update_step, init_carry, jnp.arange(n_steps)
    )

    return growth_evolution


def _save_growth_evolution(growth_evolution, params):
    output_file = my_files.OutputFile('growth', '.pkl', params)
    data_handler = my_files.DataHandler(output_file)
    data_handler.save(growth_evolution)


@my_utils.timer
def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    init_vertices = jax_arrays['init_vertices']
    all_cells = init_vertices[jax_arrays['indices']]

    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])
    init_aspect_ratios = my_utils.calc_aspect_ratios(
        all_cells, jax_arrays['valid_mask']
    )
    optimal_angles = my_utils.calc_optimal_angles(jax_arrays['valid_mask'])

    goal_areas = (
        params.numerical['max_area_scaling'] * init_areas.mean()
    )
    goal_aspect_ratios = 0.5 * jnp.ones_like(init_areas)

    growth_evolution = iterate(
        goal_areas, goal_aspect_ratios, init_areas, init_aspect_ratios,
        optimal_angles, params.numerical['n_growth_steps'], jax_arrays,
        params.numerical
    )

    _save_growth_evolution(growth_evolution, params)


if __name__ == '__main__':
    _main()
