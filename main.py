import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import growth, my_files, my_utils


def _calc_scaling(min_scaling, max_scaling, logits):
    scaling = (
        min_scaling + (max_scaling - min_scaling) * jax.nn.sigmoid(logits)
    )
    return scaling


def _calc_area_scaling(max_scaling, logits):
    area_scaling = _calc_scaling(0.0, max_scaling, logits)
    return area_scaling


def _calc_aspect_ratio_scaling(logits):
    aspect_ratio_scaling = _calc_scaling(0.0, 1.0, logits)
    return aspect_ratio_scaling


def _calc_goal_areas(init_areas, max_area_scaling, logits):
    goal_areas = init_areas.mean() * _calc_area_scaling(
        max_area_scaling, logits
    )
    return goal_areas


def _calc_goal_aspect_ratios(as_logits):
    goal_aspect_ratios = _calc_aspect_ratio_scaling(as_logits)
    return goal_aspect_ratios


def _calc_shape_loss(final_vertices, boundary_mask, outer_shape):
    diff_vectors = final_vertices[:,None] - outer_shape
    dists = jnp.linalg.norm(diff_vectors, axis=2)
    min_sq_dists = jnp.min(dists**2, axis=1)
    shape_loss = jnp.sum(min_sq_dists * boundary_mask)
    return shape_loss


class _MyOptimizer:
    def __init__(self, ar_logits, as_logits):
        self._lr_schedule = optax.cosine_decay_schedule(
            init_value=0.02, decay_steps=500, alpha=0.1
        )

        self._optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale_by_schedule(self._lr_schedule),
            optax.scale(-1.0)
        )
        self._state = self._optimizer.init(
            params=(ar_logits, as_logits)
        )

    def update(self, ar_logits, as_logits, ar_grads, as_grads):
        updates, self._state = self._optimizer.update(
            (ar_grads, as_grads), self._state
        )
        ar_logits, as_logits = optax.apply_updates(
            (ar_logits, as_logits), updates
        )
        return ar_logits, as_logits


def _save_output_params(init_centroids, init_areas, best_goal_areas_scalings,
                        best_goal_areas, best_goal_aspect_ratios, params):
    param_dict = {
        'init_centroid_x': init_centroids[:,0],
        'init_centroid_y': init_centroids[:,1],
        'init_area': init_areas,
        'goal_area_scaling': best_goal_areas_scalings,
        'goal_area': best_goal_areas,
        'goal_aspect_ratio': best_goal_aspect_ratios
    }

    df = pd.DataFrame(param_dict)

    output_file = my_files.OutputFile(
        'output_params', '.txt', params
    ).get_path()
    df.to_csv(output_file, sep='\t', index=True, header=True)


def _iterate_towards_shape(jax_arrays, all_params):
    params = all_params.numerical

    init_vertices = jax_arrays['init_vertices']
    all_cells = init_vertices[jax_arrays['indices']]
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    def shape_loss_func(ar_logits, as_logits):
        goal_areas = _calc_goal_areas(
            init_areas, params['max_area_scaling'], ar_logits
        )
        goal_aspect_ratios = _calc_goal_aspect_ratios(as_logits)

        growth_evolution = growth.iterate(
            goal_areas, goal_aspect_ratios, jax_arrays, params
        )
        final_vertices = growth_evolution[-1]

        shape_loss = _calc_shape_loss(
            final_vertices, jax_arrays['boundary_mask'],
            jax_arrays['outer_shape']
        )

        return shape_loss, final_vertices

    val_grad_loss = jax.jit(
        jax.value_and_grad(shape_loss_func, has_aux=True, argnums=(0, 1))
    )

    # Initialize parameters
    ar_logits = jnp.zeros_like(init_areas)
    as_logits = jnp.zeros_like(init_areas)

    optimizer = _MyOptimizer(ar_logits, as_logits)

    figure = my_utils.Figure(init_vertices)

    final_tissues_dir = my_files.OutputDir('final_tissues', all_params)

    shape_loss = jnp.inf

    for shape_step in range(params['n_shape_steps']):
        (new_shape_loss, final_vertices), (ar_grads, as_grads) = (
            val_grad_loss(ar_logits, as_logits)
        )
        ar_logits, as_logits = (
            optimizer.update(ar_logits, as_logits, ar_grads, as_grads)
        )

        print(f'{shape_step}: Shape loss = {new_shape_loss}')

        if new_shape_loss < shape_loss:
            best_goal_areas_scalings = _calc_area_scaling(
                params['max_area_scaling'], ar_logits
            )
            best_goal_areas = _calc_goal_areas(
                init_areas, params['max_area_scaling'], ar_logits
            )
            best_goal_aspect_ratios = _calc_goal_aspect_ratios(as_logits)

            shape_loss = new_shape_loss

            print(f'(Stored params with new best shape loss.)')
            print('')

        if shape_step % 100 == 0:
            figure.plot(
                final_tissues_dir.get_path(), final_vertices, jax_arrays,
                shape_step
            )

    _save_output_params(
        jax_arrays['init_centroids'], init_areas, best_goal_areas_scalings,
        best_goal_areas, best_goal_aspect_ratios, all_params
    )


@my_utils.timer
def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    _iterate_towards_shape(jax_arrays, params)


if __name__ == '__main__':
    _main()
