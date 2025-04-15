import jax
import jax.numpy as jnp

import numpy as np
import optax

import growth, init_systems, utils


_N_SHAPE_STEPS = 2000
_N_GROWTH_STEPS = 200
_GROWTH_LEARNING_RATE = 0.015

_AREAS_LOSS_WEIGHT = 1.0
_ANGLES_LOSS_WEIGHT = 1.0
_ASPECT_RATIO_LOSS_WEIGHT = 1.0
_OPTIMAL_ASPECT_RATIO = 1/3.0

_GOAL_AREA_WEIGHT = 1e-5


def _sigmoid(variations):
    return 1.0 + 1.5 * jax.nn.sigmoid(variations)


def _calc_aspect_ratio_scales(jax_arrays):
    basal_mask = jax_arrays['basal_mask']
    aspect_ratio_scales = np.ones(len(basal_mask))
    aspect_ratio_scales[basal_mask] /= _OPTIMAL_ASPECT_RATIO
    return jnp.array(aspect_ratio_scales)


def _calc_goal_areas(init_areas, aspect_ratio_scales, variations):
    goal_areas = init_areas * aspect_ratio_scales * _sigmoid(variations)
    return goal_areas


def _calc_shape_loss(final_vertices, boundary_mask, outer_shape):
    diff_vectors = final_vertices[:,None] - outer_shape
    dists = jnp.linalg.norm(diff_vectors, axis=2)
    min_sq_dists = jnp.min(dists**2, axis=1)
    shape_loss = jnp.sum(min_sq_dists * boundary_mask)
    return shape_loss


def _iterate_towards_shape(jax_arrays, outer_shape):
    init_vertices = jax_arrays['init_vertices']
    all_cells = init_vertices[jax_arrays['indices']]
    init_areas = growth.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    aspect_ratio_scales = _calc_aspect_ratio_scales(jax_arrays)

    growth_params = {
        'goal_area_weight': _GOAL_AREA_WEIGHT,
        'learning_rate': _GROWTH_LEARNING_RATE,
        'n_steps': _N_GROWTH_STEPS,
        'areas_loss_weight': _AREAS_LOSS_WEIGHT,
        'angles_loss_weight': _ANGLES_LOSS_WEIGHT,
        'aspect_ratio_loss_weight': _ASPECT_RATIO_LOSS_WEIGHT,
        'optimal_aspect_ratio': _OPTIMAL_ASPECT_RATIO
    }

    def shape_loss_func(variations):
        goal_areas = _calc_goal_areas(
            init_areas, aspect_ratio_scales, variations
        )
        final_vertices = growth.iterate(goal_areas, jax_arrays, growth_params)
        shape_loss = _calc_shape_loss(
            final_vertices, jax_arrays['boundary_mask'], outer_shape
        )

        return shape_loss, final_vertices

    val_grad_loss = jax.jit(jax.value_and_grad(shape_loss_func, has_aux=True))

    variations = jnp.zeros_like(init_areas)

    figure = utils.Figure(init_vertices)

    init_learning_rate = 0.001
    optimizer = optax.adam(init_learning_rate)
    opt_state = optimizer.init(params=variations)

    output_dirs = utils.get_output_dirs()
    for shape_step in range(_N_SHAPE_STEPS):
        (shape_loss, final_vertices), grads = val_grad_loss(variations)
        updates, opt_state = optimizer.update(grads, opt_state)
        variations = optax.apply_updates(variations, updates)
        print(f'{shape_step}: Shape loss = {shape_loss}')

        if shape_step % 100 == 0:
            figure.plot(
                output_dirs['final_tissues'], final_vertices, jax_arrays,
                outer_shape, shape_step
            )

    print(f'Best final goal area scalings {_sigmoid(variations)}')
    final_goal_areas = _calc_goal_areas(
        init_areas, aspect_ratio_scales, variations
    )
    growth.iterate_and_plot(
        output_dirs['best_growth'], final_goal_areas, outer_shape, jax_arrays,
        growth_params
    )


@utils.timer
def _main():
    np.random.seed(0)
    jax.config.update('jax_enable_x64', True)

    utils.make_output_dirs()

    args = utils.parse_args()

    polygons = init_systems.get_polygons(args)

    jax_arrays = utils.get_jax_arrays(polygons)

    outer_shape = utils.make_ellipse(args.init_system)

    _iterate_towards_shape(jax_arrays, outer_shape)


if __name__ == "__main__":
    _main()
