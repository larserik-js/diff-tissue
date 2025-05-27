import jax
import jax.numpy as jnp
import numpy as np
import optax

import growth, init_systems, my_utils


def _sigmoid(max_area_scaling, variations):
    return 1.0 + (max_area_scaling - 1.0) * jax.nn.sigmoid(variations)


def _calc_aspect_ratio_scales(jax_arrays, optimal_aspect_ratio):
    basal_mask = jax_arrays['basal_mask']
    aspect_ratio_scales = np.ones(len(basal_mask))
    aspect_ratio_scales[basal_mask] /= optimal_aspect_ratio
    return jnp.array(aspect_ratio_scales)


def _calc_goal_areas(init_areas, max_area_scaling, aspect_ratio_scales,
                     variations):
    goal_areas = init_areas * aspect_ratio_scales * _sigmoid(
        max_area_scaling, variations
    )
    return goal_areas


def _calc_goal_aspect_ratios(as_variations):
    return jax.nn.sigmoid(as_variations)


def _calc_shape_loss(final_vertices, boundary_mask, outer_shape):
    diff_vectors = final_vertices[:,None] - outer_shape
    dists = jnp.linalg.norm(diff_vectors, axis=2)
    min_sq_dists = jnp.min(dists**2, axis=1)
    shape_loss = jnp.sum(min_sq_dists * boundary_mask)
    return shape_loss


def _iterate_towards_shape(jax_arrays, params, output_dirs):
    init_vertices = jax_arrays['init_vertices']
    all_cells = init_vertices[jax_arrays['indices']]
    init_areas = growth.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    aspect_ratio_scales = _calc_aspect_ratio_scales(
        jax_arrays, params['optimal_aspect_ratio']
    )

    def shape_loss_func(ar_variations, as_variations):
        goal_areas = _calc_goal_areas(
            init_areas, params['max_area_scaling'], aspect_ratio_scales,
            ar_variations
        )
        goal_aspect_ratios = _calc_goal_aspect_ratios(as_variations)

        final_vertices = growth.iterate(
            goal_areas, goal_aspect_ratios, jax_arrays, params
        )
        shape_loss = _calc_shape_loss(
            final_vertices, jax_arrays['boundary_mask'],
            jax_arrays['outer_shape']
        )

        return shape_loss, final_vertices

    val_grad_loss = jax.jit(
        jax.value_and_grad(shape_loss_func, has_aux=True, argnums=(0, 1))
    )

    ar_variations = jnp.zeros_like(init_areas)
    as_variations = jnp.zeros_like(init_areas)

    figure = my_utils.Figure(init_vertices)

    init_learning_rate = 0.01
    optimizer = optax.adam(init_learning_rate)
    opt_state = optimizer.init(params=(ar_variations, as_variations))

    for shape_step in range(params['n_shape_steps']):
        (shape_loss, final_vertices), (ar_grads, as_grads) = (
            val_grad_loss(ar_variations, as_variations)
        )
        updates, opt_state = optimizer.update((ar_grads, as_grads), opt_state)
        ar_variations, as_variations = optax.apply_updates(
            (ar_variations, as_variations), updates
        )
        print(f'{shape_step}: Shape loss = {shape_loss}')

        if shape_step % 100 == 0:
            figure.plot(
                output_dirs['final_tissues'], final_vertices, jax_arrays,
                shape_step
            )

    print(f'Best final goal area scalings: {_sigmoid(
        params['max_area_scaling'], ar_variations)}'
    )
    print(
        'Best final goal aspect ratios: '
        f'{_calc_goal_aspect_ratios(as_variations)}'
    )
    final_goal_areas = _calc_goal_areas(
        init_areas, params['max_area_scaling'], aspect_ratio_scales,
        ar_variations
    )
    final_goal_aspect_ratios = _calc_goal_aspect_ratios(as_variations)
    growth.iterate_and_plot(
        output_dirs['best_growth'], final_goal_areas, final_goal_aspect_ratios,
        jax_arrays, params
    )


@my_utils.timer
def _main():
    np.random.seed(0)
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()
    output_dirs = my_utils.OutputDirs(['final_tissues', 'best_growth'], params)
    output_dirs.make()

    factory = init_systems.get_factory(params.shape, params.system)
    polygons = factory.get_polygons()
    outer_shape = factory.get_outer_shape()

    jax_arrays = my_utils.get_jax_arrays(polygons, outer_shape)

    _iterate_towards_shape(jax_arrays, params.numerical, output_dirs.get())


if __name__ == "__main__":
    _main()
