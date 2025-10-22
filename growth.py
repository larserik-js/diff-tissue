import jax
import jax.numpy as jnp
import numpy as np

import my_files, my_utils


def _calc_optimal_angles(valid_mask):
    n_vertices = valid_mask.sum(axis=1) - 2
    interior_angles = (n_vertices - 2) * jnp.pi / n_vertices
    optimal_angles = jnp.pi - interior_angles
    optimal_angles = optimal_angles[:, None]
    return optimal_angles


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


@jax.jit
def _update_targets(init_targets, goals, t_frac):
    targets = (
        init_targets + (goals - init_targets) * jnp.sin(0.5 * jnp.pi * t_frac)
    )
    return targets


def _update_vertices(vertices, t, init_areas, goal_areas, init_aspect_ratios,
                     goal_aspect_ratios, optimal_angles, jax_arrays,
                     calc_loss_and_grads, params):
    t_frac = t / params['n_growth_steps']
    target_areas = _update_targets(init_areas, goal_areas, t_frac)
    target_aspect_ratios = _update_targets(
        init_aspect_ratios, goal_aspect_ratios, t_frac
    )
    _, grads = calc_loss_and_grads(
        vertices, target_areas, target_aspect_ratios, optimal_angles,
        jax_arrays, params
    )
    vertices -= (
        params['growth_learning_rate'] * grads * jax_arrays['free_mask']
    )

    return vertices


def iterate(goal_areas, goal_aspect_ratios, areas, aspect_ratios, jax_arrays,
            params):
    optimal_angles = _calc_optimal_angles(jax_arrays['valid_mask'])

    _calc_loss_and_grads = jax.jit(jax.value_and_grad(_calc_growth_loss))

    def update_step(carry, t):
        (vertices, areas, aspect_ratios, goal_areas,
         goal_aspect_ratios) = carry

        vertices = _update_vertices(
            vertices, t, areas, goal_areas, aspect_ratios, goal_aspect_ratios,
            optimal_angles, jax_arrays, _calc_loss_and_grads, params
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
        update_step, init_carry, jnp.arange(params['n_growth_steps'])
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

    goal_areas = (
        params.numerical['max_area_scaling'] * init_areas.mean()
    )
    goal_aspect_ratios = 0.5 * jnp.ones_like(init_areas)

    growth_evolution = iterate(
        goal_areas, goal_aspect_ratios, init_areas, init_aspect_ratios,
        jax_arrays, params.numerical
    )

    _save_growth_evolution(growth_evolution, params)


if __name__ == '__main__':
    _main()
