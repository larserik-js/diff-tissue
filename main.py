import argparse
import jax
import jax.numpy as jnp
import os
from pathlib import Path
import timeit

import matplotlib.pyplot as plt
import numpy as np
import optax

import init_systems


_N_SHAPE_STEPS = 100
_N_GROWTH_STEPS = 500
_LEARNING_RATE = 1e-4

_AREAS_LOSS_WEIGHT = 10.0
_ANGLES_LOSS_WEIGHT = 5.0
_ASPECT_RATIO_LOSS_WEIGHT = 100.0
_OPTIMAL_ASPECT_RATIO = 1/8

_GOAL_AREA_WEIGHT = 5e-5


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--init_system',
        type=str,
        choices=['full', 'simple', 'voronoi'],
        default='simple',
        help='Initial polygon configuration.'
    )
    return parser.parse_args()


def _get_device():
    return jax.devices('cpu')[0]


def _get_project_dir():
    project_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    return project_dir


def _get_output_dirs():
    project_dir = _get_project_dir()
    output_dirs = {'final_tissues': project_dir / 'final_tissues',
                   'best_growth': project_dir / 'best_growth'}
    return output_dirs


def _make_output_dirs():
    output_dirs = _get_output_dirs()
    for output_dir in output_dirs.values():
        output_dir.mkdir(exist_ok=True)


def _timer(func):
    def timed(*args, **kwargs):
        t_init = timeit.default_timer()
        res = func(*args, **kwargs)
        t_end = timeit.default_timer()

        t_tot = t_end - t_init

        print(f'Total time: {t_tot:.4f} s')
        return res

    return timed


def _send_to_device(jax_arrays):
    return jax.device_put(jax_arrays, device=_get_device())


def _get_jax_arrays(polygons):
    arrays = {
        'init_vertices': polygons.get_vertices(),
        'indices': polygons.get_polygon_inds(),
        'valid_mask': polygons.get_valid_mask(),
        'fixed_mask': polygons.get_fixed_mask(),
        'basal_mask': polygons.get_basal_mask(),
        'boundary_mask': polygons.get_boundary_mask()
    }
    jax_arrays = {k: _send_to_device(jnp.array(v)) for k, v in arrays.items()}

    return jax_arrays


@jax.jit
def _update_target_areas(target_areas, t, goal_areas):
    w = _GOAL_AREA_WEIGHT * t
    target_areas = (1 - w) * target_areas + w * goal_areas
    return target_areas


def _calc_optimal_angles(mask):
    n_vertices = mask.sum(axis=1) - 2
    interior_angles = (n_vertices - 2) * jnp.pi / n_vertices
    optimal_angles = jnp.pi - interior_angles
    optimal_angles = optimal_angles[:, None]
    return optimal_angles


def _calc_all_areas(all_cells, valid_mask):
    xs = all_cells[:, 1:-1, 0]
    y_plus_ones = all_cells[:, 2:, 1]
    y_minus_ones = all_cells[:, :-2, 1]

    valid = valid_mask[:, 1:-1] & valid_mask[:, 2:] & valid_mask[:, :-2]

    first_term = xs * y_plus_ones
    first_term = jnp.sum(first_term * valid, axis=1)
    second_term = xs * y_minus_ones
    second_term = jnp.sum(second_term * valid, axis=1)

    # Abs. because vertex orientation can be
    # both clockwise and counter-clockwise
    areas = 0.5 * jnp.abs(first_term - second_term)

    return areas


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


def _calc_aspect_ratios_loss(aspect_ratios, basal_mask):
    aspect_ratio_diffs = aspect_ratios - _OPTIMAL_ASPECT_RATIO
    aspect_ratios_loss = _ASPECT_RATIO_LOSS_WEIGHT * jnp.sum(
        jnp.square(basal_mask * aspect_ratio_diffs)
    )
    return aspect_ratios_loss


def _calc_growth_loss(vertices, target_areas, optimal_angles, jax_arrays):
    all_cells = vertices[jax_arrays['indices']]
    areas = _calc_all_areas(all_cells, jax_arrays['valid_mask'])
    # aspect_ratios = _calc_aspect_ratios(all_cells, jax_arrays['valid_mask'])

    areas_loss = _AREAS_LOSS_WEIGHT * jnp.sum((target_areas - areas)**2)
    angles_loss = _ANGLES_LOSS_WEIGHT * _calc_all_angles_loss(
        all_cells, jax_arrays['valid_mask'], optimal_angles
    )
    # aspect_ratios_loss = _calc_aspect_ratios_loss(
    #     aspect_ratios, jax_arrays['basal_mask']
    # )

    # loss = areas_loss + angles_loss + aspect_ratios_loss
    loss = areas_loss + angles_loss

    return loss


def _get_ax_lims(vertices):
    minvals = vertices.min(axis=0)
    maxvals = vertices.max(axis=0)
    center = (minvals + maxvals) / 2
    dims = maxvals - minvals
    xlim = center + jnp.array([-1.0, 1.0]) * dims[0]
    ylim = center + jnp.array([-1.0, 1.0]) * dims[1]
    return xlim, ylim


def _format(ax, ax_lims):
    ax.clear()
    ax.set_xlim(ax_lims[0])
    ax.set_ylim(ax_lims[1])
    ax.set_aspect('equal')


def _add_artists(ax, vertices, jax_arrays, outer_shape):
    indices = jax_arrays['indices']
    for i in range(indices.shape[0]):
        vertex_inds = indices[i][jax_arrays['valid_mask'][i]]
        polygon = vertices[vertex_inds]
        ax.scatter(polygon[:, 0], polygon[:, 1], s=2.0, color='green', zorder=1)
        ax.plot(polygon[:, 0], polygon[:, 1], lw=0.7, color='black', zorder=2)

    base_y = 18.635
    ax.plot([-20, 10], [base_y, base_y], 'k', lw=0.7)
    ax.plot([70, 100], [base_y, base_y], 'k', lw=0.7)

    boundary_vertices = vertices[jax_arrays['boundary_mask']]
    ax.scatter(
        boundary_vertices[:, 0], boundary_vertices[:, 1], s=20.0, color='g',
        marker='s', zorder=3
    )

    ax.plot(
        outer_shape[:, 0], outer_shape[:, 1], 'ro-', markersize=3,
        label='Outer shape'
    )


def _save_figure(fig, output_dir, step):
    fig_path = output_dir / f'step={step}.png'
    fig.savefig(fig_path, dpi=100)


def _plot(fig, ax, ax_lims, output_dir, vertices, jax_arrays, outer_shape,
          step):
    _format(ax, ax_lims)
    _add_artists(ax, vertices, jax_arrays, outer_shape)
    _save_figure(fig, output_dir, step)


def _iterate_over_growth(goal_areas, jax_arrays):
    all_cells = jax_arrays['init_vertices'][jax_arrays['indices']]
    target_areas = _calc_all_areas(all_cells, jax_arrays['valid_mask'])
    optimal_angles = _calc_optimal_angles(jax_arrays['valid_mask'])

    _calc_loss_and_grads = jax.value_and_grad(_calc_growth_loss)
    _calc_loss_and_grads = jax.jit(_calc_loss_and_grads)

    def update_step(carry, t):
        vertices, target_areas = carry

        target_areas = _update_target_areas(target_areas, t, goal_areas)
        _, grads = _calc_loss_and_grads(
            vertices, target_areas, optimal_angles, jax_arrays
        )
        vertices -= _LEARNING_RATE * grads * jax_arrays['fixed_mask']

        return (vertices, target_areas), None

    init_carry = (jax_arrays['init_vertices'], target_areas)
    final_carry, _ = jax.lax.scan(
        update_step, init_carry, jnp.arange(_N_GROWTH_STEPS)
    )
    final_vertices, target_areas = final_carry

    return final_vertices


def _sigmoid(variations):
    return 1.0 + 2.0 * jax.nn.sigmoid(variations)


def _make_ellipse(num_points=50, a=10.0, b=15.0, center=(40.0, 40.0)):
    angles = jnp.linspace(0, 2 * jnp.pi, num_points, endpoint=True)
    x = center[0] + a * jnp.cos(angles)
    y = center[1] + b * jnp.sin(angles)
    return jnp.stack([x, y], axis=1)


def _calc_shape_loss(final_vertices, boundary_mask, outer_shape):
    diff_vectors = final_vertices[:,None] - outer_shape
    dists = jnp.linalg.norm(diff_vectors, axis=2)
    min_dists = jnp.min(dists, axis=1)
    shape_loss = jnp.sum(min_dists * boundary_mask)
    return shape_loss


def _make_growth_plots(final_variations, init_areas, jax_arrays, outer_shape):
    goal_areas = init_areas * _sigmoid(final_variations)
    vertices = jax_arrays['init_vertices']
    all_cells = vertices[jax_arrays['indices']]
    target_areas = _calc_all_areas(all_cells, jax_arrays['valid_mask'])
    optimal_angles = _calc_optimal_angles(jax_arrays['valid_mask'])

    _calc_loss_and_grads = jax.value_and_grad(_calc_growth_loss)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax_lims = _get_ax_lims(vertices)
    output_dir = _get_output_dirs()['best_growth']
    _plot(
        fig, ax, ax_lims, output_dir, vertices, jax_arrays, outer_shape,
        step=0
    )

    for t in jnp.arange(_N_GROWTH_STEPS):
        target_areas = _update_target_areas(target_areas, t, goal_areas)
        _, grads = _calc_loss_and_grads(
            vertices, target_areas, optimal_angles, jax_arrays
        )
        vertices -= _LEARNING_RATE * grads * jax_arrays['fixed_mask']

        if t % 5 == 0:
            _plot(
                fig, ax, ax_lims, output_dir, vertices, jax_arrays, outer_shape,
                step=t+1
            )


def _iterate_towards_shape(jax_arrays):
    init_vertices = jax_arrays['init_vertices']
    all_cells = init_vertices[jax_arrays['indices']]
    init_areas = _calc_all_areas(all_cells, jax_arrays['valid_mask'])

    outer_shape = _make_ellipse()

    def shape_loss_func(variations):
        goal_areas = init_areas * _sigmoid(variations)
        final_vertices = _iterate_over_growth(goal_areas, jax_arrays)
        shape_loss = _calc_shape_loss(
            final_vertices, jax_arrays['boundary_mask'], outer_shape
        )

        return shape_loss, final_vertices

    val_grad_loss = jax.jit(jax.value_and_grad(shape_loss_func, has_aux=True))

    variations = jnp.zeros_like(init_areas)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax_lims = _get_ax_lims(init_vertices)

    init_learning_rate = 0.01
    optimizer = optax.adam(init_learning_rate)
    opt_state = optimizer.init(params=variations)

    output_dir = _get_output_dirs()['final_tissues']
    for shape_step in range(_N_SHAPE_STEPS):
        (shape_loss, final_vertices), grads = val_grad_loss(variations)
        updates, opt_state = optimizer.update(grads, opt_state)
        variations = optax.apply_updates(variations, updates)
        print(f'{shape_step}: Shape loss = {shape_loss}')

        _plot(
            fig, ax, ax_lims, output_dir, final_vertices, jax_arrays,
            outer_shape, shape_step
        )

    print(f'Best final goal area scalings {_sigmoid(variations)}')
    _make_growth_plots(variations, init_areas, jax_arrays, outer_shape)


@_timer
def _main():
    np.random.seed(0)
    jax.config.update('jax_enable_x64', True)

    _make_output_dirs()

    args = _parse_args()

    polygons = init_systems.get_polygons(args)

    jax_arrays = _get_jax_arrays(polygons)

    _iterate_towards_shape(jax_arrays)


if __name__ == "__main__":
    _main()
