import argparse
import jax
import jax.numpy as jnp
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import init_systems


_N_TIMESTEPS = 100_000
_LEARNING_RATE = 1e-4

_AREAS_LOSS_WEIGHT = 10.0
_ANGLES_LOSS_WEIGHT = 50.0
_ASPECT_RATIO_LOSS_WEIGHT = 100.0
_OPTIMAL_ASPECT_RATIO = 1/8


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--init_system',
        type=str,
        choices=['full', 'voronoi'],
        default='full',
        help='Initial polygon configuration.'
    )
    return parser.parse_args()


def _get_device():
    return jax.devices('cpu')[0]


def _get_project_dir():
    project_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    return project_dir


def _get_output_dir():
    project_dir = _get_project_dir()
    output_dir = project_dir / 'output'
    return output_dir


def _make_output_dir():
    output_dir = _get_output_dir()
    output_dir.mkdir(exist_ok=True, parents=True)


def _send_to_device(jax_arrays):
    return jax.device_put(jax_arrays, device=_get_device())


def _get_jax_arrays(polygons):
    vertices = jnp.array(polygons.get_vertices())
    indices = jnp.array(polygons.get_polygon_inds())
    valid_mask = jnp.array(polygons.get_valid_mask())
    fixed_mask = jnp.array(polygons.get_fixed_mask())
    basal_mask = jnp.array(polygons.get_basal_mask())

    jax_arrays = (vertices, indices, valid_mask, fixed_mask, basal_mask)
    jax_arrays = _send_to_device(jax_arrays)

    return jax_arrays


@jax.jit
def _update_target_areas(target_areas):
    areas_scales = 0.00001 * jnp.ones_like(target_areas)
    # Crude assumption that inner cells have vertices in the first half
    # of the vertex array
    inner_cells_scale = 1.0
    areas_scales = areas_scales.at[:int(0.5 * target_areas.shape[0])].mul(
        inner_cells_scale
    )
    target_areas += areas_scales * target_areas
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


def _calc_loss(vertices, indices, valid_mask, target_areas, optimal_angles,
               basal_mask):
    all_cells = vertices[indices]
    areas = _calc_all_areas(all_cells, valid_mask)
    aspect_ratios = _calc_aspect_ratios(all_cells, valid_mask)

    areas_loss = _AREAS_LOSS_WEIGHT * jnp.sum((target_areas - areas)**2)
    angles_loss = _ANGLES_LOSS_WEIGHT * _calc_all_angles_loss(
        all_cells, valid_mask, optimal_angles
    )
    aspect_ratios_loss = _calc_aspect_ratios_loss(aspect_ratios, basal_mask)

    jax.debug.print('Areas loss: {}', areas_loss)
    jax.debug.print('Angles loss: {}', angles_loss)
    jax.debug.print('Aspect ratio loss: {}', aspect_ratios_loss)
    jax.debug.print('')

    loss = areas_loss + angles_loss + aspect_ratios_loss

    return loss


def _get_ax_lims(vertices):
    minvals = vertices.min(axis=0)
    maxvals = vertices.max(axis=0)
    center = (minvals + maxvals) / 2
    dims = maxvals - minvals
    xlim = center + jnp.array([-1.0, 1.0]) * dims[0]
    ylim = center + jnp.array([-1.0, 1.0]) * dims[1]
    return xlim, ylim


def _format(ax, xlim, ylim):
    ax.clear()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')


def _plot(ax, vertices, indices, valid_mask):
    for i in range(indices.shape[0]):
        vertex_inds = indices[i][valid_mask[i]]
        polygon = vertices[vertex_inds]
        ax.scatter(polygon[:, 0], polygon[:, 1], s=2.0, color='green', zorder=1)
        ax.plot(polygon[:, 0], polygon[:, 1], lw=0.7, color='black', zorder=2)

    base_y = 18.635
    ax.plot([-20, 10], [base_y, base_y], 'k', lw=0.7)
    ax.plot([70, 100], [base_y, base_y], 'k', lw=0.7)


def _save_figure(fig, step):
    output_dir = _get_output_dir()
    fig_path = output_dir / f'step_{step}.png'
    
    fig.savefig(fig_path, dpi=100)


def _iterate(vertices, indices, valid_mask, fixed_mask, basal_mask):
    fig, ax = plt.subplots(figsize=(10, 10))
    xlim, ylim = _get_ax_lims(vertices)

    all_cells = vertices[indices]
    target_areas = _calc_all_areas(all_cells, valid_mask)
    optimal_angles = _calc_optimal_angles(valid_mask)

    _calc_loss_and_grads = jax.value_and_grad(_calc_loss)
    _calc_loss_and_grads = jax.jit(_calc_loss_and_grads)

    for t in range(_N_TIMESTEPS):
        target_areas = _update_target_areas(target_areas)
        loss, grads = _calc_loss_and_grads(
            vertices, indices, valid_mask, target_areas, optimal_angles,
            basal_mask
        )
        vertices -= _LEARNING_RATE * grads * fixed_mask

        if t % int(1000) == 0:
            print(f'Step {t}, Loss: {loss}')
        
            _format(ax, xlim, ylim)
            _plot(ax, vertices, indices, valid_mask)
            _save_figure(fig, t)


def _main():
    np.random.seed(0)
    jax.config.update('jax_enable_x64', True)

    _make_output_dir()

    args = parse_args()

    polygons = init_systems.get_polygons(args)

    vertices, indices, valid_mask, fixed_mask, basal_mask = _get_jax_arrays(
        polygons
    )

    _iterate(vertices, indices, valid_mask, fixed_mask, basal_mask)


if __name__ == "__main__":
    _main()
