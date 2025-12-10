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


def _calc_smoothing_stds(logits):
    smoothing_stds = _calc_sigmoid(0.0, 10.0, logits)
    return smoothing_stds


def _calc_inverse_smoothing_stds(smoothing_stds):
    logits = _calc_inverse_sigmoid(0.0, 10.0, smoothing_stds)
    return logits


def _calc_goal_areas(init_areas, min_area_scaling, max_area_scaling, logits):
    goal_areas = init_areas.mean() * _calc_area_scaling(
        min_area_scaling, max_area_scaling, logits
    )
    return goal_areas


def _calc_goal_aspect_ratios(as_logits):
    goal_aspect_ratios = _calc_sigmoid(-1.0, 1.0, as_logits)
    return goal_aspect_ratios


def _calc_knot_weights(std_logits, dist_vecs):
    smoothing_stds = _calc_smoothing_stds(std_logits)
    knot_weights = jnp.exp(
        -jnp.sum(dist_vecs**2 / (2 * smoothing_stds[None,None,:]**2), axis=2)
    )
    knot_weights += 1e-8
    knot_weights = knot_weights / jnp.sum(knot_weights, axis=1)[:, None]

    return knot_weights


def _knots_to_full_shape(lc_goals, n_left_logits, weights):
    right_goals = lc_goals[:n_left_logits]
    all_goals = jnp.concatenate([lc_goals, right_goals])
    all_goals = jnp.sum(all_goals[None, :] * weights, axis=1)
    return all_goals


def _calc_shape_loss(final_vertices, boundary_mask, outer_shape, min_dist_mask):
    diff_vectors = final_vertices[:,None] - outer_shape
    dists = jnp.linalg.norm(diff_vectors, axis=2)
    dists_cubed = dists**3
    masked_dists = jnp.where(min_dist_mask, dists_cubed, jnp.inf)
    min_cubed_dists = jnp.min(masked_dists, axis=1)
    shape_loss = jnp.sum(min_cubed_dists * boundary_mask)
    return shape_loss


def _loss_f(lc_ar_logits, lc_as_logits, std_logits, n_left_logits, dist_vecs,
            init_areas, min_dist_mask, n_growth_steps, jax_arrays, params):

    min_area_scaling = 1 / params['growth_scale']
    lc_goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params['max_area_scaling'], lc_ar_logits
    )
    lc_goal_aspect_ratios = _calc_goal_aspect_ratios(lc_as_logits)

    knot_weights = _calc_knot_weights(std_logits, dist_vecs)

    goal_areas = _knots_to_full_shape(
        lc_goal_areas, n_left_logits, knot_weights
    )
    goal_aspect_ratios = _knots_to_full_shape(
        lc_goal_aspect_ratios, n_left_logits, knot_weights
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
    jax.value_and_grad(_loss_f, has_aux=True, argnums=(0, 1, 2)),
    static_argnames=['n_left_logits', 'n_growth_steps']
)


def _find_closest_vertices(left_knots, mapped_vertices):
    dist_vecs = left_knots[:,None,:] - mapped_vertices
    dists = jnp.linalg.norm(dist_vecs, axis=2)
    closest_vertices = jnp.argmin(dists, axis=1)
    return closest_vertices


def _find_closest_polygons(left_knots, mapped_vertices, vertex_polygons):
    closest_vertices = _find_closest_vertices(
        left_knots, mapped_vertices
    )
    closest_polygons = vertex_polygons[closest_vertices]
    return closest_polygons


def _calc_mean_closest_metric(mapped_metrics, closest_polygons):
    closest_metrics = mapped_metrics[closest_polygons]
    closest_metrics_with_nans = jnp.where(
        closest_polygons != -1, closest_metrics, jnp.nan
    )
    mean_mapped_metrics = jnp.nanmean(closest_metrics_with_nans, axis=1)
    return mean_mapped_metrics


def _get_knots_area_scalings(closest_polygons, mapped_areas, init_areas):
    mean_mapped_areas = _calc_mean_closest_metric(
        mapped_areas, closest_polygons
    )
    mean_init_areas = _calc_mean_closest_metric(init_areas, closest_polygons)
    knots_area_scalings = mean_mapped_areas / mean_init_areas
    return knots_area_scalings


def _calc_logits(knots, params, vertex_polygons, mapped_vertices,
                 init_areas, mapped_areas, mapped_aspect_ratios):
    closest_polygons = _find_closest_polygons(
        knots, mapped_vertices, vertex_polygons
    )

    area_scalings = _get_knots_area_scalings(
        closest_polygons, mapped_areas, init_areas
    )
    min_area_scaling = 1 / params.numerical['growth_scale']
    max_area_scaling = params.numerical['max_area_scaling']
    area_scalings = area_scalings.clip(
        min_area_scaling + 1e-8, max_area_scaling - 1e-8
    )

    aspect_ratios = _calc_mean_closest_metric(
        mapped_aspect_ratios, closest_polygons
    )
    ar_logits = _calc_inverse_area_scaling(
        min_area_scaling, max_area_scaling, area_scalings
    )
    as_logits = _calc_inverse_aspect_ratios(aspect_ratios)

    return ar_logits, as_logits


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
    mapped_aspect_ratios = my_utils.calc_aspect_ratios(
        all_mapped_cells, jax_arrays['valid_mask']
    )

    init_logits = dict()
    knot_positions = ['left', 'center']
    for pos in knot_positions:
        knots = jax_arrays[pos + '_knots']
        ar_logits, as_logits = _calc_logits(
            knots, params, jax_arrays['vertex_polygons'], mapped_vertices,
            init_areas, mapped_areas, mapped_aspect_ratios
        )
        init_logits[pos + '_area_scalings'] = ar_logits
        init_logits[pos + '_aspect_ratios'] = as_logits

    init_logits['smoothing_stds'] = _calc_inverse_smoothing_stds(jnp.ones(2))
    return init_logits


class _MyOptimizer:
    def __init__(self, ar_logits, as_logits, std_logits):
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
            params=(ar_logits, as_logits, std_logits)
        )

    def update(self, ar_logits, as_logits, std_logits, ar_grads, as_grads,
               std_grads):
        updates, self._state = self._optimizer.update(
            (ar_grads, as_grads, std_grads), self._state
        )
        ar_logits, as_logits, std_logits = optax.apply_updates(
            (ar_logits, as_logits, std_logits), updates
        )
        return ar_logits, as_logits, std_logits


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


def _iterate_towards_shape(init_logits, jax_arrays, all_params):
    params = all_params.numerical

    vertices = jax_arrays['init_vertices']
    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    min_area_scaling = 1 / params['growth_scale']

    mapped_centroids = _get_mapped_centroids(jax_arrays)
    knots = jax_arrays['all_knots']
    dist_vecs = mapped_centroids[:, None] - knots[None, :]

    min_dist_mask = _make_min_dist_mask(jax_arrays)

    # Initialize parameters
    lc_ar_logits = jnp.concatenate(
        [init_logits['left_area_scalings'], init_logits['center_area_scalings']]
    )
    lc_as_logits = jnp.concatenate(
        [init_logits['left_aspect_ratios'], init_logits['center_aspect_ratios']]
    )
    n_left_logits = len(init_logits['left_area_scalings'])
    std_logits = init_logits['smoothing_stds']

    optimizer = _MyOptimizer(lc_ar_logits, lc_as_logits, std_logits)

    best_loss = jnp.inf

    final_tissues = jnp.empty(
        (params['n_shape_steps'] + 1, vertices.shape[0], vertices.shape[1])
    )
    final_tissues = final_tissues.at[0].set(vertices)

    for shape_step in range(params['n_shape_steps']):
        (loss, vertices), (lc_ar_grads, lc_as_grads, std_grads) = (
            _calc_loss_val_grads(
                lc_ar_logits, lc_as_logits, std_logits, n_left_logits,
                dist_vecs, init_areas, min_dist_mask, params['n_growth_steps'],
                jax_arrays, params
            )
        )
        lc_ar_logits, lc_as_logits, std_logits = (
            optimizer.update(
                lc_ar_logits, lc_as_logits, std_logits, lc_ar_grads,
                lc_as_grads, std_grads
            )
        )

        print(f'{shape_step}: Shape loss = {loss}')

        if loss < best_loss:
            best_goal_area_scalings = _calc_area_scaling(
                min_area_scaling, params['max_area_scaling'], lc_ar_logits
            )
            best_goal_areas = _calc_goal_areas(
                init_areas, min_area_scaling, params['max_area_scaling'],
                lc_ar_logits
            )
            best_goal_aspect_ratios = _calc_goal_aspect_ratios(lc_as_logits)

            best_knot_weights = _calc_knot_weights(std_logits, dist_vecs)

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
        best_goal_area_scalings, n_left_logits, best_knot_weights
    )
    best_goal_areas = _knots_to_full_shape(
        best_goal_areas, n_left_logits, best_knot_weights
    )
    best_goal_aspect_ratios = _knots_to_full_shape(
        best_goal_aspect_ratios, n_left_logits, best_knot_weights
    )

    final_knot_weights = _calc_knot_weights(std_logits, dist_vecs)
    final_goal_areas_scalings = _calc_area_scaling(
        min_area_scaling, params['max_area_scaling'], lc_ar_logits
    )
    final_goal_areas_scalings = _knots_to_full_shape(
        final_goal_areas_scalings, n_left_logits, final_knot_weights
    )
    final_goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params['max_area_scaling'], lc_ar_logits
    )
    final_goal_areas = _knots_to_full_shape(
        final_goal_areas, n_left_logits, final_knot_weights
    )
    final_goal_aspect_ratios = _calc_goal_aspect_ratios(lc_as_logits)
    final_goal_aspect_ratios = _knots_to_full_shape(
        final_goal_aspect_ratios, n_left_logits, final_knot_weights
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

    init_logits = _get_init_logits(jax_arrays, params)

    best_loss, final_tissues, tabular_output = _iterate_towards_shape(
        init_logits, jax_arrays, params
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
