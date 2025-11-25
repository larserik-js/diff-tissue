import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import diffeomorphism, init_systems, morph, my_files, my_utils


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


def _knots_to_full_shape(left_goals, weights):
    symmetric_goals = jnp.tile(left_goals, 2)
    all_goals = np.sum(symmetric_goals[None, :] * weights, axis=1)
    return all_goals


def _loss_f(ar_logits, as_logits, knot_weights, init_areas,
            min_dist_mask, n_growth_steps, jax_arrays, params):

    min_area_scaling = 1 / params['growth_scale']
    left_goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params['max_area_scaling'], ar_logits
    )
    left_goal_aspect_ratios = _calc_goal_aspect_ratios(as_logits)

    goal_areas = _knots_to_full_shape(left_goal_areas, knot_weights)
    goal_aspect_ratios = _knots_to_full_shape(
        left_goal_aspect_ratios, knot_weights
    )

    growth_evolution = morph.iterate(
        goal_areas, goal_aspect_ratios, n_growth_steps, jax_arrays, params
    )
    final_vertices = growth_evolution[-1]

    loss = _calc_shape_loss(
        final_vertices, jax_arrays['boundary_mask'],
        jax_arrays['outer_shape'], min_dist_mask
    )

    return loss, final_vertices


_calc_loss_val_grads = jax.jit(
    jax.value_and_grad(_loss_f, has_aux=True, argnums=(0, 1)),
    static_argnames=['n_growth_steps']
)


def _get_init_logits(left_knot_inds, jax_arrays, params):
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
        min_area_scaling + 1e-8, max_area_scaling - 1e-8
    )

    mapped_aspect_ratios = my_utils.calc_aspect_ratios(
        all_mapped_cells, jax_arrays['valid_mask']
    )

    left_area_scalings = mapped_area_scalings[left_knot_inds]
    left_aspect_ratios = mapped_aspect_ratios[left_knot_inds]

    init_logits = {
        'left_area_scalings': _calc_inverse_area_scaling(
            min_area_scaling, max_area_scaling, left_area_scalings
        ),
        'left_aspect_ratios': _calc_inverse_aspect_ratios(left_aspect_ratios)
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


def _save_final_tissues(final_tissues, params):
    output_file = my_files.OutputFile('final_tissues', '.pkl', params)
    data_handler = my_files.DataHandler(output_file)
    data_handler.save(final_tissues)


def _save_output_params(param_dict, params):
    df = pd.DataFrame(param_dict)
    output_file = my_files.get_output_params_file(params)
    df.to_csv(output_file, sep='\t', index=True, header=True)


def _make_min_dist_mask(jax_arrays):
    min_dist_mask = jnp.ones(
        (jax_arrays['init_vertices'].shape[0],
         jax_arrays['outer_shape'].shape[0]),
         dtype=bool
    )
    fixed_mask = jnp.any(~jax_arrays['free_mask'], axis=1)

    outer_shape_basal_mask = jnp.isclose(
        jax_arrays['outer_shape'][:,1], init_systems.Coords.base_origin[1]
    )
    min_dist_mask = min_dist_mask.at[fixed_mask].set(outer_shape_basal_mask)
    return min_dist_mask


def _get_mapped_centroids(jax_arrays):
    mapped_vertices = diffeomorphism.get_mapped_vertices(jax_arrays)
    mapped_centroids = my_utils.calc_centroids(
        mapped_vertices, jax_arrays['indices'], jax_arrays['valid_mask']
    )
    return mapped_centroids


def _flip_around_shape_y(array):
    flipped_array = jnp.empty_like(array)
    shape_x = init_systems.Coords.shape_origin[0]
    flipped_array = flipped_array.at[:,0].set(2 * shape_x - array[:,0])
    flipped_array = flipped_array.at[:,1].set(array[:,1])
    return flipped_array


def _get_symmetric_knots(left_knots):
    right_knots = _flip_around_shape_y(left_knots)
    symmetric_knots = jnp.concatenate([left_knots, right_knots], axis=0)
    return symmetric_knots


def _calc_knot_weights(mapped_centroids, knots):
    dist_vecs = (mapped_centroids[:, None] - knots[None, :])
    sy = 1.0
    knot_weights = jnp.exp(-dist_vecs[:, :, 1]**2 / (2 * sy**2))
    knot_weights += 1e-8
    knot_weights = knot_weights / jnp.sum(knot_weights, axis=1)[:, None]

    return knot_weights


def _iterate_towards_shape(left_knot_inds, init_logits, jax_arrays, all_params):
    params = all_params.numerical

    vertices = jax_arrays['init_vertices']
    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    min_area_scaling = 1 / params['growth_scale']

    mapped_centroids = _get_mapped_centroids(jax_arrays)
    left_knots = mapped_centroids[left_knot_inds]
    knots = _get_symmetric_knots(left_knots)

    knot_weights = _calc_knot_weights(mapped_centroids, knots)

    min_dist_mask = _make_min_dist_mask(jax_arrays)

    # Initialize parameters
    ar_logits = init_logits['left_area_scalings']
    as_logits = init_logits['left_aspect_ratios']

    optimizer = _MyOptimizer(ar_logits, as_logits)

    best_loss = jnp.inf

    final_tissues = jnp.empty(
        (params['n_shape_steps'] + 1, vertices.shape[0], vertices.shape[1])
    )
    final_tissues = final_tissues.at[0].set(vertices)

    for shape_step in range(params['n_shape_steps']):
        (loss, vertices), (ar_grads, as_grads) = (
            _calc_loss_val_grads(
                ar_logits, as_logits, knot_weights, init_areas,
                min_dist_mask, params['n_growth_steps'], jax_arrays, params
            )
        )
        ar_logits, as_logits = (
            optimizer.update(ar_logits, as_logits, ar_grads, as_grads)
        )

        print(f'{shape_step}: Shape loss = {loss}')

        if loss < best_loss:
            best_goal_area_scalings = _calc_area_scaling(
                min_area_scaling, params['max_area_scaling'], ar_logits
            )
            best_goal_areas = _calc_goal_areas(
                init_areas, min_area_scaling, params['max_area_scaling'],
                ar_logits
            )
            best_goal_aspect_ratios = _calc_goal_aspect_ratios(as_logits)

            best_loss = loss

            print(f'(Stored params with new best loss.)')
            print('')

        final_tissues = final_tissues.at[shape_step+1].set(vertices)

    # Calculate output params
    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
    final_areas = my_utils.calc_all_areas(
        all_cells, jax_arrays['valid_mask']
    )
    final_aspect_ratios = my_utils.calc_aspect_ratios(
        all_cells, jax_arrays['valid_mask']
    )
    best_goal_area_scalings = _knots_to_full_shape(
        best_goal_area_scalings, knot_weights
    )
    best_goal_areas = _knots_to_full_shape(best_goal_areas, knot_weights)
    best_goal_aspect_ratios = _knots_to_full_shape(
        best_goal_aspect_ratios, knot_weights
    )

    final_goal_areas_scalings = _calc_area_scaling(
        min_area_scaling, params['max_area_scaling'], ar_logits
    )
    final_goal_areas_scalings = _knots_to_full_shape(
        final_goal_areas_scalings, knot_weights
    )
    final_goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params['max_area_scaling'], ar_logits
    )
    final_goal_areas = _knots_to_full_shape(
        final_goal_areas, knot_weights
    )
    final_goal_aspect_ratios = _calc_goal_aspect_ratios(as_logits)
    final_goal_aspect_ratios = _knots_to_full_shape(
        final_goal_aspect_ratios, knot_weights
    )

    tabular_output = {
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

    return best_loss, final_tissues, tabular_output


def _run(params):
    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    left_knot_inds = jnp.array([89, 60, 25, 54, 62, 11, 22])
    init_logits = _get_init_logits(left_knot_inds, jax_arrays, params)

    best_loss, final_tissues, tabular_output = _iterate_towards_shape(
        left_knot_inds, init_logits, jax_arrays, params
    )

    return best_loss, final_tissues, tabular_output


@my_utils.timer
def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()
    _, final_tissues, tabular_output = _run(params)
    
    _save_final_tissues(final_tissues, params)
    _save_output_params(tabular_output, params)


if __name__ == '__main__':
    _main()
