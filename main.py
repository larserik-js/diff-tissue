import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import diffeomorphism, morph, init_systems, my_files, my_utils, plotting


def _calc_sigmoid(min_val, max_val, logits):
    sigmoid_vals = min_val + (max_val - min_val) * jax.nn.sigmoid(logits)
    return sigmoid_vals


def _calc_inverse_sigmoid(min_val, max_val, sigmoid_vals):
    s = (sigmoid_vals - min_val) / (max_val - min_val)
    logits = jnp.log(s / (1 - s))
    return logits


def _calc_area_scaling(min_scaling, max_scaling, logits):
    area_scaling = _calc_sigmoid(min_scaling, max_scaling, logits)
    return area_scaling


def _calc_inverse_area_scaling(min_scaling, max_scaling, area_scalings):
    logits = _calc_inverse_sigmoid(min_scaling, max_scaling, area_scalings)
    return logits


def _calc_inverse_aspect_ratios(aspect_ratios):
    logits = _calc_inverse_sigmoid(-1.0, 1.0, aspect_ratios)
    return logits


def _calc_goal_areas(init_areas, min_area_scaling, max_area_scaling, logits):
    goal_areas = init_areas.mean() * _calc_area_scaling(
        min_area_scaling, max_area_scaling, logits
    )
    return goal_areas


def _calc_goal_aspect_ratios(as_logits):
    goal_aspect_ratios = _calc_sigmoid(-1.0, 1.0, as_logits)
    return goal_aspect_ratios


def _calc_shape_loss(final_vertices, boundary_mask, outer_shape, min_dist_mask):
    diff_vectors = final_vertices[:,None] - outer_shape
    dists = jnp.linalg.norm(diff_vectors, axis=2)
    dists_cubed = dists**3
    masked_dists = jnp.where(min_dist_mask, dists_cubed, jnp.inf)
    min_cubed_dists = jnp.min(masked_dists, axis=1)
    shape_loss = jnp.sum(min_cubed_dists * boundary_mask)
    return shape_loss


def _calc_area_regularization_loss(final_vertices, jax_arrays):
    all_cells = my_utils.get_all_cells(final_vertices, jax_arrays['indices'])
    final_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])
    area_reg_loss = jnp.var(final_areas)
    return area_reg_loss


def _shape_loss_f(ar_logits, as_logits, init_areas, min_dist_mask,
                  n_growth_steps, jax_arrays, params):
    min_area_scaling = 1 / params['growth_scale']
    goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params['max_area_scaling'], ar_logits
    )
    goal_aspect_ratios = _calc_goal_aspect_ratios(as_logits)

    growth_evolution = morph.iterate(
        goal_areas, goal_aspect_ratios, n_growth_steps, jax_arrays, params
    )
    final_vertices = growth_evolution[-1]

    shape_loss = _calc_shape_loss(
        final_vertices, jax_arrays['boundary_mask'],
        jax_arrays['outer_shape'], min_dist_mask
    )

    area_reg_loss = 5.0 * _calc_area_regularization_loss(
        final_vertices, jax_arrays
    )

    shape_loss += area_reg_loss

    return shape_loss, final_vertices


_calc_shape_loss_val_grads = jax.jit(
    jax.value_and_grad(_shape_loss_f, has_aux=True, argnums=(0, 1)),
    static_argnames=['n_growth_steps']
)


def _get_init_logits(jax_arrays, params):
    mapped_vertices = diffeomorphism.get_mapped_vertices(jax_arrays)
    all_mapped_cells = mapped_vertices[jax_arrays['indices']]
    mapped_areas = my_utils.calc_all_areas(
        all_mapped_cells, jax_arrays['valid_mask']
    )

    all_cells = my_utils.get_all_cells(
        jax_arrays['init_vertices'], jax_arrays['indices']
    )
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    mapped_area_scalings = mapped_areas / init_areas

    min_area_scaling = 1 / params.numerical['growth_scale']
    max_area_scaling = params.numerical['max_area_scaling']
    mapped_area_scalings = mapped_area_scalings.clip(
        min_area_scaling, max_area_scaling
    )

    mapped_aspect_ratios = my_utils.calc_aspect_ratios(
        all_mapped_cells, jax_arrays['valid_mask']
    )

    init_logits = {
        'area_scalings': _calc_inverse_area_scaling(
            min_area_scaling, max_area_scaling, mapped_area_scalings
        ),
        'aspect_ratios': _calc_inverse_aspect_ratios(mapped_aspect_ratios)
    }
    return init_logits


class _MyOptimizer:
    def __init__(self, ar_logits, as_logits):
        self._lr_schedule = optax.cosine_decay_schedule(
            init_value=0.02, decay_steps=500, alpha=1e-5
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


def _save_output_params(param_dict, params):
    df = pd.DataFrame(param_dict)
    output_file = my_files.get_output_params_file(params)
    df.to_csv(output_file, sep='\t', index=True, header=True)


def _make_min_dist_mask(jax_arrays):
    min_dist_mask = np.ones(
        (jax_arrays['init_vertices'].shape[0],
         jax_arrays['outer_shape'].shape[0]),
         dtype=bool
    )
    fixed_mask = jnp.any(~jax_arrays['free_mask'], axis=1)

    outer_shape_basal_mask = np.isclose(
        jax_arrays['outer_shape'][:,1], init_systems.Coords.base_origin[1]
    )
    min_dist_mask[fixed_mask] = outer_shape_basal_mask
    return jnp.array(min_dist_mask)


def _iterate_towards_shape(init_logits, jax_arrays, all_params):
    params = all_params.numerical

    vertices = jax_arrays['init_vertices']
    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    min_area_scaling = 1 / params['growth_scale']

    # Initialize parameters
    ar_logits = init_logits['area_scalings']
    as_logits = init_logits['aspect_ratios']

    min_dist_mask = _make_min_dist_mask(jax_arrays)

    optimizer = _MyOptimizer(ar_logits, as_logits)

    final_tissues_dir = my_files.OutputDir('final_tissues', all_params)

    figure = plotting.MorphFigure(final_tissues_dir.path, jax_arrays)

    shape_loss = jnp.inf

    for shape_step in range(params['n_shape_steps']):
        (new_shape_loss, vertices), (ar_grads, as_grads) = (
            _calc_shape_loss_val_grads(
                ar_logits, as_logits, init_areas, min_dist_mask,
                params['n_growth_steps'], jax_arrays, params
            )
        )
        ar_logits, as_logits = (
            optimizer.update(ar_logits, as_logits, ar_grads, as_grads)
        )

        print(f'{shape_step}: Shape loss = {new_shape_loss}')

        if new_shape_loss < shape_loss:
            best_goal_area_scalings = _calc_area_scaling(
                min_area_scaling, params['max_area_scaling'], ar_logits
            )
            best_goal_areas = _calc_goal_areas(
                init_areas, min_area_scaling, params['max_area_scaling'],
                ar_logits
            )
            best_goal_aspect_ratios = _calc_goal_aspect_ratios(as_logits)

            shape_loss = new_shape_loss

            print(f'(Stored params with new best shape loss.)')
            print('')

        if shape_step % 10 == 0:
            figure.save_plot(vertices, shape_step)
    figure.save_plot(vertices, shape_step)

    # Calculate output params
    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
    final_areas = my_utils.calc_all_areas(
        all_cells, jax_arrays['valid_mask']
    )
    final_aspect_ratios = my_utils.calc_aspect_ratios(
        all_cells, jax_arrays['valid_mask']
    )
    final_goal_areas_scalings = _calc_area_scaling(
        min_area_scaling, params['max_area_scaling'], ar_logits
    )
    final_goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params['max_area_scaling'], ar_logits
    )
    final_goal_aspect_ratios = _calc_goal_aspect_ratios(as_logits)

    param_dict = {
        'init_area': init_areas,
        'final_area': final_areas,
        'final_aspect_ratio': final_aspect_ratios,
        'best_goal_area_scaling': best_goal_area_scalings,
        'best_goal_area': best_goal_areas,
        'best_goal_aspect_ratio': best_goal_aspect_ratios,
        'final_goal_area_scaling': final_goal_areas_scalings,
        'final_goal_area': final_goal_areas,
        'final_goal_aspect_ratio': final_goal_aspect_ratios
    }

    _save_output_params(param_dict, all_params)


@my_utils.timer
def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    init_logits = _get_init_logits(jax_arrays, params)

    _iterate_towards_shape(init_logits, jax_arrays, params)


if __name__ == '__main__':
    _main()
