from dataclasses import dataclass

import numpy as np
import optax

from .jax_bootstrap import jax, jnp, struct
from . import init_systems, morphing, my_utils


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


def _calc_inverse_elongations(elongations):
    logits = _calc_inverse_sigmoid(-1.0, 1.0, elongations)
    return logits


def _calc_smoothing_stds(logits):
    smoothing_stds = _calc_sigmoid(0.0, 10.0, logits)
    return smoothing_stds


def _calc_inverse_smoothing_stds(smoothing_stds):
    logits = _calc_inverse_sigmoid(0.0, 10.0, smoothing_stds)
    return logits


def _calc_goal_areas_(init_areas, min_area_scaling, max_area_scaling, logits):
    goal_areas = init_areas.mean() * _calc_area_scaling(
        min_area_scaling, max_area_scaling, logits
    )
    return goal_areas


def _knots_to_full_shape(lc_goals, n_left_logits, weights):
    right_goals = lc_goals[:n_left_logits]
    all_goals = jnp.concatenate([lc_goals, right_goals])
    all_goals = jnp.sum(all_goals[None, :] * weights, axis=1)
    return all_goals


def _calc_goal_areas(
        init_areas, min_area_scaling, max_area_scaling, ar_logits,
        proximal_mask, knots, knot_ctx
    ):
    goal_areas = _calc_goal_areas_(
        init_areas, min_area_scaling, max_area_scaling, ar_logits
    )

    if knots:
        goal_areas = _knots_to_full_shape(
            goal_areas, knot_ctx.n_left_logits, knot_ctx.knot_weights
        )

    # Multiplies the areas of proximal polygons.
    # This is necessary to accompany the elongation.
    goal_areas = jnp.where(proximal_mask, 2.0 * goal_areas, goal_areas)

    return goal_areas


def _calc_goal_elongations_(el_logits):
    goal_elongations = _calc_sigmoid(-1.0, 1.0, el_logits)
    return goal_elongations


def _calc_goal_elongations(el_logits, proximal_mask, knots, knot_ctx):
    goal_elongations = _calc_goal_elongations_(el_logits)

    if knots:
        goal_elongations = _knots_to_full_shape(
            goal_elongations, knot_ctx.n_left_logits, knot_ctx.knot_weights
        )

    # Multiplies the elongations of proximal polygons.
    # The actual constant does not carry physical meaning at this point.
    goal_elongations = jnp.where(
        proximal_mask, 3.0 * goal_elongations, goal_elongations
    )

    return goal_elongations


def _calc_knot_weights(std_logits, dist_vecs):
    smoothing_stds = _calc_smoothing_stds(std_logits)
    knot_weights = jnp.exp(
        -jnp.sum(dist_vecs**2 / (2 * smoothing_stds[None,None,:]**2), axis=2)
    )
    knot_weights += 1e-8
    knot_weights = knot_weights / jnp.sum(knot_weights, axis=1)[:, None]

    return knot_weights


def _calc_shape_loss(final_vertices, boundary_mask, outer_shape, min_dist_mask):
    diff_vectors = final_vertices[:,None] - outer_shape
    dists = jnp.linalg.norm(diff_vectors, axis=2)
    dists_cubed = dists**3
    masked_dists = jnp.where(min_dist_mask, dists_cubed, jnp.inf)
    min_cubed_dists = jnp.min(masked_dists, axis=1)
    shape_loss = jnp.sum(min_cubed_dists * boundary_mask)
    return shape_loss


def _loss_f(ar_logits, el_logits, knot_ctx, init_areas, min_dist_mask,
            n_growth_steps, jax_arrays, params):
    min_area_scaling = 1 / params.growth_scale
    goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params.max_area_scaling, ar_logits,
        jax_arrays['proximal_mask'], False, knot_ctx
    )
    goal_elongations = _calc_goal_elongations(
        el_logits, jax_arrays['proximal_mask'], False, knot_ctx
    )

    growth_evolution = morphing.iterate(
        goal_areas, goal_elongations, n_growth_steps, jax_arrays, params
    )
    final_vertices = growth_evolution[-1]

    loss = _calc_shape_loss(
        final_vertices, jax_arrays['boundary_mask'], jax_arrays['outer_shape'],
        min_dist_mask
    )
    aux_data = (final_vertices, knot_ctx)

    return loss, aux_data


def _loss_f_knots(
        ar_logits, el_logits, std_logits, knot_ctx, init_areas, min_dist_mask,
        n_growth_steps, jax_arrays, params
    ):
    updated_knot_weights = _calc_knot_weights(std_logits, knot_ctx.dist_vecs)
    knot_ctx = knot_ctx.replace(knot_weights=updated_knot_weights)

    min_area_scaling = 1 / params.growth_scale
    goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params.max_area_scaling, ar_logits,
        jax_arrays['proximal_mask'], True, knot_ctx
    )
    goal_elongations = _calc_goal_elongations(
        el_logits, jax_arrays['proximal_mask'], True, knot_ctx
    )

    growth_evolution = morphing.iterate(
        goal_areas, goal_elongations, n_growth_steps, jax_arrays, params
    )
    final_vertices = growth_evolution[-1]

    loss = _calc_shape_loss(
        final_vertices, jax_arrays['boundary_mask'], jax_arrays['outer_shape'],
        min_dist_mask
    )
    aux_data = (final_vertices, knot_ctx)

    return loss, aux_data


_calc_loss_val_grads = jax.jit(
    jax.value_and_grad(_loss_f, has_aux=True, argnums=(0, 1)),
    static_argnames=['n_growth_steps']
)


_calc_loss_val_grads_knots = jax.jit(
    jax.value_and_grad(_loss_f_knots, has_aux=True, argnums=(0, 1, 2)),
    static_argnames=['n_growth_steps']
)

def _calc_knots_to_mapped_centroids_dist_vecs(knots, jax_arrays):
    dist_vecs = jax_arrays['mapped_centroids'][:, None] - knots[None, :]
    return dist_vecs


def _find_closest_polygon_by_knots(knots, jax_arrays):
    dist_vecs = _calc_knots_to_mapped_centroids_dist_vecs(knots, jax_arrays)
    dists = jnp.linalg.norm(dist_vecs, axis=2)
    closest_inds = jnp.argmin(dists, axis=0)
    return closest_inds


def _calc_mean_closest_metric(metrics, closest_poly_idx_by_knots, jax_arrays):
    """
    Calculate the mean metric of a polygon and its immediate neighbors
    for each knot.

    Args:
        metrics (jnp.ndarray): First array of shape (M,).
        closest_poly_idx_by_knots (jnp.ndarray): Second array of shape (N,).
        jax_arrays (dict): Jax arrays.

    Returns:
        jnp.ndarray: The mean metric for each knot. Array of shape (N,).
    """
    mean_closest_metrics = np.empty(closest_poly_idx_by_knots.shape[0])
    for i, poly_ind in enumerate(closest_poly_idx_by_knots):
        poly_neighbors = jax_arrays['poly_neighbors'][poly_ind]
        valid_neighbors = poly_neighbors[poly_neighbors != -1]
        metrics_ = list(metrics[valid_neighbors]) # Metrics of neighbors
        metrics_.append(metrics[poly_ind]) # Metric of polygon
        mean_closest_metrics[i] = np.mean(metrics_)
    return mean_closest_metrics


def _get_knots_area_scalings(mapped_areas, closest_poly_idx_by_knots,
                             init_areas, jax_arrays):
    mean_mapped_areas = _calc_mean_closest_metric(
        mapped_areas, closest_poly_idx_by_knots, jax_arrays
    )
    mean_init_areas = _calc_mean_closest_metric(
        init_areas, closest_poly_idx_by_knots, jax_arrays
    )
    knots_area_scalings = mean_mapped_areas / mean_init_areas
    return knots_area_scalings


def _calc_std_logits(jax_arrays):
    knots_x_diff = (
        jax_arrays['center_knots'][0,0] - jax_arrays['left_knots'][-1,0]
    )
    knots_y_diff = (
        jax_arrays['left_knots'][-1,1] - jax_arrays['left_knots'][-2,1]
    )

    init_smoothing_stds = jnp.array([knots_x_diff, knots_y_diff])
    std_logits = _calc_inverse_smoothing_stds(init_smoothing_stds)
    return std_logits


def _calc_logits(params, area_scalings, elongations):
    min_area_scaling = 1 / params.growth_scale
    max_area_scaling = params.max_area_scaling
    area_scalings = area_scalings.clip(
        min_area_scaling + 1e-8, max_area_scaling - 1e-8
    )
    ar_logits = _calc_inverse_area_scaling(
        min_area_scaling, max_area_scaling, area_scalings
    )
    el_logits = _calc_inverse_elongations(elongations)

    return ar_logits, el_logits


def _get_poly_init_logits(params, mapped_areas, mapped_elongations, init_areas):
    mapped_area_scalings = mapped_areas / init_areas

    ar_logits, el_logits = _calc_logits(
        params, mapped_area_scalings, mapped_elongations
    )
    init_logits = (ar_logits, el_logits)
    return init_logits


def _get_knot_init_logits(jax_arrays, params, mapped_areas, mapped_elongations,
                          init_areas):
    knot_positions = ['left', 'center']
    left_and_center_ar_logits = []
    left_and_center_el_logits = []
    for pos in knot_positions:
        knots = jax_arrays[pos + '_knots']
        closest_polygon_by_knots = _find_closest_polygon_by_knots(
            knots, jax_arrays
        )

        area_scalings = _get_knots_area_scalings(
            mapped_areas, closest_polygon_by_knots, init_areas, jax_arrays
        )
        elongations = _calc_mean_closest_metric(
            mapped_elongations, closest_polygon_by_knots, jax_arrays
        )
        ar_logits, el_logits = _calc_logits(
            params, area_scalings, elongations
        )
        left_and_center_ar_logits.append(ar_logits)
        left_and_center_el_logits.append(el_logits)

    ar_logits = jnp.concatenate(left_and_center_ar_logits)
    el_logits = jnp.concatenate(left_and_center_el_logits)
    std_logits = _calc_std_logits(jax_arrays)

    init_logits = (ar_logits, el_logits, std_logits)
    return init_logits


def _get_init_logits(jax_arrays, params):
    all_mapped_cells = my_utils.get_all_cells(
        jax_arrays['mapped_vertices'], jax_arrays['indices']
    )
    mapped_areas = my_utils.calc_all_areas(
        all_mapped_cells, jax_arrays['valid_mask']
    )
    all_cells = my_utils.get_all_cells(
        jax_arrays['init_vertices'], jax_arrays['indices']
    )
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])
    mapped_elongations = my_utils.calc_elongations(
        all_mapped_cells, jax_arrays['valid_mask']
    )

    if params.knots:
        init_logits = _get_knot_init_logits(
            jax_arrays, params, mapped_areas, mapped_elongations, init_areas
        )
    else:
        init_logits = _get_poly_init_logits(
            params, mapped_areas, mapped_elongations, init_areas
        )
    return init_logits


class _MyOptimizer:
    def __init__(self, init_logits):
        self._lr_schedule = optax.cosine_decay_schedule(
            init_value=0.02, decay_steps=500, alpha=1e-5
        )
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale_by_schedule(self._lr_schedule),
            optax.scale(-1.0)
        )
        self._state = self._optimizer.init(params=init_logits)

    def update(self, logits, grads):
        updates, self._state = self._optimizer.update(grads, self._state)
        logits = optax.apply_updates(logits, updates)
        return logits


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


@struct.dataclass
class _KnotCtx:
    n_left_logits: int = struct.field(pytree_node=False)
    dist_vecs: jnp.ndarray
    knot_weights: jnp.ndarray


def _get_knot_ctx(knots, jax_arrays):
    if knots:
        dist_vecs = _calc_knots_to_mapped_centroids_dist_vecs(
            jax_arrays['all_knots'], jax_arrays
        )
        knot_ctx = _KnotCtx(
            n_left_logits = jax_arrays['left_knots'].shape[0],
            dist_vecs = dist_vecs,
            knot_weights = jnp.array([])
        )
    else:
        knot_ctx = None

    return knot_ctx


@dataclass
class _BestState:
    loss: float
    goal_areas: jnp.ndarray
    goal_elongations: jnp.ndarray
    final_areas: jnp.ndarray
    final_elongations: jnp.ndarray


def _assemble_tabular_output(init_areas, best):
    tabular_output = {
        'init_area': init_areas,
        'best_goal_area': best.goal_areas,
        'best_goal_elongation': best.goal_elongations,
        'final_area': best.final_areas,
        'final_elongation': best.final_elongations,
    }
    return tabular_output


def _iterate_towards_shape(logits, jax_arrays, params):
    vertices = jax_arrays['init_vertices']
    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    min_area_scaling = 1 / params.growth_scale

    min_dist_mask = _make_min_dist_mask(jax_arrays)

    knot_ctx = _get_knot_ctx(params.knots, jax_arrays)

    optimizer = _MyOptimizer(logits)

    best = _BestState(
        loss = jnp.inf,
        goal_areas = jnp.array([]),
        goal_elongations = jnp.array([]),
        final_areas = jnp.array([]),
        final_elongations = jnp.array([])
    )

    final_tissues = [vertices]

    steps_since_best_loss = 0

    for shape_step in range(params.n_shape_steps):
        if params.knots:
            (loss, aux_data), grads = (
                _calc_loss_val_grads_knots(
                    *logits, knot_ctx, init_areas, min_dist_mask,
                    params.n_growth_steps, jax_arrays, params
                )
            )
        else:
            (loss, aux_data), grads = (
                _calc_loss_val_grads(
                    *logits, knot_ctx, init_areas, min_dist_mask,
                    params.n_growth_steps, jax_arrays, params
                )
            )
        vertices, knot_ctx = aux_data

        if not params.quiet:
            print(f'{shape_step}: Shape loss = {loss}')

        ar_logits, el_logits = logits[:2]

        if loss < best.loss:
            best.loss = loss
            steps_since_best_loss = 0

            best.goal_areas = _calc_goal_areas(
                init_areas, min_area_scaling, params.max_area_scaling,
                ar_logits, jax_arrays['proximal_mask'], params.knots,
                knot_ctx
            )
            best.goal_elongations = _calc_goal_elongations(
                el_logits, jax_arrays['proximal_mask'], params.knots,
                knot_ctx
            )

            all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
            best.final_areas = my_utils.calc_all_areas(
                all_cells, jax_arrays['valid_mask']
            )
            best.final_elongations = my_utils.calc_elongations(
                all_cells, jax_arrays['valid_mask']
            )

            if not params.quiet:
                print(f'(Stored params with new best loss.)')
                print('')
        else:
            steps_since_best_loss += 1

        final_tissues.append(vertices)

        if steps_since_best_loss >= 20 and best.loss != jnp.inf:
            if not params.quiet:
                print(f'(Stopped - iteration diverged.)')
                print('')
            break
        else:
            logits = optimizer.update(logits, grads)

    final_tissues = jnp.array(final_tissues)

    tabular_output = _assemble_tabular_output(init_areas, best)

    return best.loss, final_tissues, tabular_output


def run(params):
    np.random.seed(params.seed)

    jax_arrays = my_utils.get_jax_arrays(params)

    init_logits = _get_init_logits(jax_arrays, params)

    best_loss, final_tissues, tabular_output = _iterate_towards_shape(
        init_logits, jax_arrays, params
    )

    return best_loss, final_tissues, tabular_output
