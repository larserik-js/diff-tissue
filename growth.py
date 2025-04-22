import numpy as np
import jax
import jax.numpy as jnp

import init_systems, utils


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

    aspect_ratios = widths / (heights + 1e-7)

    return aspect_ratios


def _calc_aspect_ratios_loss(aspect_ratios, basal_mask, params):
    aspect_ratio_diffs = aspect_ratios - params['optimal_aspect_ratio']
    aspect_ratios_loss = params['aspect_ratio_loss_weight'] * jnp.sum(
        jnp.square(basal_mask * aspect_ratio_diffs)
    )
    return aspect_ratios_loss


def _calc_growth_loss(vertices, target_areas, optimal_angles, jax_arrays,
                      params):
    all_cells = vertices[jax_arrays['indices']]
    areas = calc_all_areas(all_cells, jax_arrays['valid_mask'])
    aspect_ratios = _calc_aspect_ratios(all_cells, jax_arrays['valid_mask'])

    areas_loss = params['areas_loss_weight'] * jnp.sum(
        (target_areas - areas)**2
    )
    areas_loss = params['areas_loss_weight'] * jnp.sum(
        (target_areas - areas)**2
    )
    angles_loss = params['angles_loss_weight'] * _calc_all_angles_loss(
        all_cells, jax_arrays['valid_mask'], optimal_angles
    )
    aspect_ratios_loss = _calc_aspect_ratios_loss(
        aspect_ratios, jax_arrays['basal_mask'], params
    )

    loss = areas_loss + angles_loss + aspect_ratios_loss

    return loss


@jax.jit
def _update_target_areas(target_areas, t, goal_areas, goal_area_weight):
    w = goal_area_weight * t
    target_areas = (1 - w) * target_areas + w * goal_areas
    return target_areas


def iterate(goal_areas, jax_arrays, params):
    all_cells = jax_arrays['init_vertices'][jax_arrays['indices']]
    target_areas = calc_all_areas(all_cells, jax_arrays['valid_mask'])
    optimal_angles = _calc_optimal_angles(jax_arrays['valid_mask'])

    _calc_loss_and_grads = jax.value_and_grad(_calc_growth_loss)
    _calc_loss_and_grads = jax.jit(_calc_loss_and_grads)

    def update_step(carry, t):
        vertices, target_areas = carry

        target_areas = _update_target_areas(
            target_areas, t, goal_areas, params['goal_area_weight']
        )
        _, grads = _calc_loss_and_grads(
            vertices, target_areas, optimal_angles, jax_arrays, params
        )
        vertices -= params['learning_rate'] * grads * jax_arrays['fixed_mask']

        return (vertices, target_areas), None

    init_carry = (jax_arrays['init_vertices'], target_areas)
    final_carry, _ = jax.lax.scan(
        update_step, init_carry, jnp.arange(params['n_steps'])
    )
    final_vertices, target_areas = final_carry

    return final_vertices


def iterate_and_plot(output_dir, goal_areas, jax_arrays, params):
    vertices = jax_arrays['init_vertices']
    all_cells = vertices[jax_arrays['indices']]
    target_areas = calc_all_areas(all_cells, jax_arrays['valid_mask'])
    optimal_angles = _calc_optimal_angles(jax_arrays['valid_mask'])

    _calc_loss_and_grads = jax.value_and_grad(_calc_growth_loss)
    figure = utils.Figure(vertices)
    figure.plot(output_dir, vertices, jax_arrays, step=0)

    for t in jnp.arange(params['n_steps']):
        target_areas = _update_target_areas(
            target_areas, t, goal_areas, params['goal_area_weight']
        )
        _, grads = _calc_loss_and_grads(
            vertices, target_areas, optimal_angles, jax_arrays, params
        )
        vertices -= params['learning_rate'] * grads * jax_arrays['fixed_mask']

        figure.plot(output_dir, vertices, jax_arrays, step=t+1)


@utils.timer
def _main():
    params = {
        'learning_rate': 0.001,
        'n_steps': 400,
        'areas_loss_weight': 10.0,
        'angles_loss_weight': 10.0,
        'aspect_ratio_loss_weight': 1.0,
        'optimal_aspect_ratio': 1.0,
        'goal_area_weight': 1e-5
    }

    utils.make_output_dirs()

    output_dir = utils.get_output_dirs()['growth']
    args = utils.parse_args()

    factory = init_systems.get_factory(args)
    polygons = factory.get_polygons()
    shape_params = factory.get_shape_params()
    outer_shape = init_systems.Ellipse(shape_params).get()

    jax_arrays = utils.get_jax_arrays(polygons, outer_shape)

    vertices = jax_arrays['init_vertices']
    all_cells = vertices[jax_arrays['indices']]
    init_areas = calc_all_areas(all_cells, jax_arrays['valid_mask'])

    aspect_ratio_scales = np.where(
        np.isclose(polygons.get_basal_mask(), 0), 1.0,
        polygons.get_basal_mask() / params['optimal_aspect_ratio']
    )
    goal_areas = 2.5 * init_areas * aspect_ratio_scales

    iterate_and_plot(output_dir, goal_areas, jax_arrays, params)


if __name__ == '__main__':
    _main()
