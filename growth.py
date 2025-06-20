import numpy as np
import jax
import jax.numpy as jnp

import my_utils


def calc_all_areas(all_cells, valid_mask):
    xs = all_cells[:, 1:-1, 0]
    y_plus_ones = all_cells[:, 2:, 1]
    y_minus_ones = all_cells[:, :-2, 1]

    valid = valid_mask[:, 1:-1] & valid_mask[:, 2:] & valid_mask[:, :-2]

    first_term = xs * y_plus_ones
    first_term = jnp.sum(first_term * valid, axis=1)
    second_term = xs * y_minus_ones
    second_term = jnp.sum(second_term * valid, axis=1)

    # Assumes vertices are ordered counter-clockwise
    areas = 0.5 * (first_term - second_term)

    return areas


def _calc_optimal_angles(mask):
    n_vertices = mask.sum(axis=1) - 2
    interior_angles = (n_vertices - 2) * jnp.pi / n_vertices
    optimal_angles = jnp.pi - interior_angles
    optimal_angles = optimal_angles[:, None]
    return optimal_angles


def _calc_all_angles_loss(all_cells, valid_mask, optimal_angles):
    epsilon = 1e-7
    edges = all_cells[:, 1:] - all_cells[:, :-1]
    dot_products = jnp.sum(edges[:, :-1] * edges[:, 1:], axis=2)
    norms = jnp.linalg.norm(edges + epsilon, axis=2)
    cosines = dot_products / (epsilon + norms[:, :-1] * norms[:, 1:])
    clip_value = 1.0 - epsilon
    cosines = jnp.clip(cosines, -clip_value, clip_value)
    angles = jnp.arccos(cosines)

    valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    valid = valid[:, 1:] & valid[:, :-1]
    angles_loss = jnp.sum((angles - optimal_angles)**2 * valid)

    return angles_loss


def _masked_min(values, mask):
    masked_values = jnp.where(mask, values, jnp.inf)
    return jnp.min(masked_values, axis=1)


def _masked_max(values, mask):
    masked_values = jnp.where(mask, values, -jnp.inf)
    return jnp.max(masked_values, axis=1)


def _calc_aspect_ratios(all_cells, valid_mask):
    min_xys = _masked_min(all_cells, valid_mask[:, :, None])
    max_xys = _masked_max(all_cells, valid_mask[:, :, None])

    widths = max_xys[:,0] - min_xys[:,0]
    heights = max_xys[:,1] - min_xys[:,1]

    aspect_ratios = widths / (heights + widths)

    return aspect_ratios


def _calc_aspect_ratios_loss(target_aspect_ratios, aspect_ratios, basal_mask):
    aspect_ratio_diffs = target_aspect_ratios - aspect_ratios
    aspect_ratios_loss = jnp.sum(jnp.square(basal_mask * aspect_ratio_diffs))
    return aspect_ratios_loss


def _calc_growth_loss(vertices, target_areas, target_aspect_ratios,
                      optimal_angles, jax_arrays, params):
    all_cells = vertices[jax_arrays['indices']]
    areas = calc_all_areas(all_cells, jax_arrays['valid_mask'])
    aspect_ratios = _calc_aspect_ratios(all_cells, jax_arrays['valid_mask'])

    areas_loss = params['areas_loss_weight'] * jnp.sum(
        (target_areas - areas)**2
    )
    angles_loss = params['angles_loss_weight'] * _calc_all_angles_loss(
        all_cells, jax_arrays['valid_mask'], optimal_angles
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
        params['growth_learning_rate'] * grads * jax_arrays['fixed_mask']
    )

    return vertices


def iterate(goal_areas, goal_aspect_ratios, jax_arrays, params):
    all_cells = jax_arrays['init_vertices'][jax_arrays['indices']]
    init_areas = calc_all_areas(all_cells, jax_arrays['valid_mask'])
    init_aspect_ratios = _calc_aspect_ratios(
        all_cells, jax_arrays['valid_mask']
    )
    optimal_angles = _calc_optimal_angles(jax_arrays['valid_mask'])

    _calc_loss_and_grads = jax.jit(jax.value_and_grad(_calc_growth_loss))

    def update_step(carry, t):
        (vertices, init_areas, init_aspect_ratios, goal_areas,
         goal_aspect_ratios) = carry

        vertices = _update_vertices(
            vertices, t, init_areas, goal_areas, init_aspect_ratios,
            goal_aspect_ratios, optimal_angles, jax_arrays,
            _calc_loss_and_grads, params
        )

        carry = (
            vertices, init_areas, init_aspect_ratios, goal_areas,
            goal_aspect_ratios
        )

        return carry, None

    init_carry = (
        jax_arrays['init_vertices'], init_areas, init_aspect_ratios,
        goal_areas, goal_aspect_ratios
    )
    final_carry, _ = jax.lax.scan(
        update_step, init_carry, jnp.arange(params['n_growth_steps'])
    )
    final_vertices, _, _, _, _ = final_carry

    return final_vertices


def iterate_and_plot(output_dir, goal_areas, goal_aspect_ratios, jax_arrays,
                     params):
    vertices = jax_arrays['init_vertices']
    all_cells = vertices[jax_arrays['indices']]
    init_areas = calc_all_areas(all_cells, jax_arrays['valid_mask'])
    init_aspect_ratios = _calc_aspect_ratios(
        all_cells, jax_arrays['valid_mask']
    )
    optimal_angles = _calc_optimal_angles(jax_arrays['valid_mask'])

    _calc_loss_and_grads = jax.value_and_grad(_calc_growth_loss)

    figure = my_utils.Figure(vertices)
    figure.plot(output_dir, vertices, jax_arrays, step=0)

    for t in jnp.arange(params['n_growth_steps']):
        vertices = _update_vertices(
            vertices, t, init_areas, goal_areas, init_aspect_ratios,
            goal_aspect_ratios, optimal_angles, jax_arrays,
            _calc_loss_and_grads, params
        )

        if (t + 1) % 2 == 0:
            figure.plot(output_dir, vertices, jax_arrays, step=t + 1)


@my_utils.timer
def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    init_vertices = jax_arrays['init_vertices']
    all_cells = init_vertices[jax_arrays['indices']]
    init_areas = calc_all_areas(all_cells, jax_arrays['valid_mask'])

    aspect_ratio_scales = my_utils.calc_aspect_ratio_scales(
        jax_arrays, params.numerical['optimal_aspect_ratio']
    )
    goal_areas = (
        params.numerical['max_area_scaling'] * init_areas.mean() *
        aspect_ratio_scales
    )
    goal_aspect_ratios = 0.5 * jnp.ones_like(init_areas)

    output_dir = my_utils.OutputDir('growth', params)
    output_dir.make()

    iterate_and_plot(
        output_dir.get_param_path(), goal_areas, goal_aspect_ratios, jax_arrays,
        params.numerical
    )


if __name__ == '__main__':
    _main()
