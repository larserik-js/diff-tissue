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


def _calc_inverse_areas(min_area, max_area, areas):
    logits = _calc_inverse_sigmoid(min_area, max_area, areas)
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


def _calc_goal_area_bounds(tutte_areas, params):
    min_goal_area = tutte_areas.min() / params.growth_scale
    max_goal_area = tutte_areas.max() * params.max_area_scaling
    return (min_goal_area, max_goal_area)


def _calc_goal_areas_(goal_area_bounds, ar_logits):
    min_goal_area, max_goal_area = goal_area_bounds
    goal_areas = _calc_sigmoid(min_goal_area, max_goal_area, ar_logits)
    return goal_areas


def _knots_to_full_shape(lc_goals, n_left_logits, weights):
    right_goals = lc_goals[:n_left_logits]
    all_goals = jnp.concatenate([lc_goals, right_goals])
    all_goals = jnp.sum(all_goals[None, :] * weights, axis=1)
    return all_goals


def _calc_goal_areas(
        goal_area_bounds, ar_logits, proximal_mask, knots, knot_ctx
    ):
    goal_areas = _calc_goal_areas_(goal_area_bounds, ar_logits)

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
        dist_vecs = _calc_knots_to_tutte_centroids_dist_vecs(
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


def _update_knot_ctx(logits, knot_ctx, knots):
    if knots:
        std_logits = logits[2]
        updated_knot_weights = _calc_knot_weights(
            std_logits, knot_ctx.dist_vecs
        )
        knot_ctx = knot_ctx.replace(knot_weights=updated_knot_weights)
    return knot_ctx


def _calc_outer_shape_segments(outer_shape):
    closed_outer_shape = jnp.concatenate([outer_shape, outer_shape[:1]], axis=0)
    segments = closed_outer_shape[1:] - closed_outer_shape[:-1]
    return segments


def _calc_shape_loss(final_vertices, boundary_mask, outer_shape, min_dist_mask):
    outer_shape_segs = _calc_outer_shape_segments(outer_shape) # (M, 2)

    # Expand dimensions for broadcasting
    # final_vertices -> (N, 1, 2)
    # outer_shape_segs -> (1, M, 2)
    final_vertices = final_vertices[:, None, :]
    outer_shape = outer_shape[None, :, :]
    outer_shape_segs = outer_shape_segs[None, :, :]

    dist_vecs = final_vertices - outer_shape # (N, M, 2)

    denom = jnp.sum(outer_shape_segs * outer_shape_segs, axis=2) # (1, M)
    t = jnp.sum(dist_vecs * outer_shape_segs, axis=2) / denom # (N, M)

    t = jax.nn.sigmoid(10.0 * (t - 0.5)) # Instead of clipping to [0, 1]

    projection = outer_shape + t[..., None] * outer_shape_segs # (N, M, 2)

    dists = jnp.linalg.norm(final_vertices - projection, axis=2) # (N, M)
    dists_squared = dists**2

    masked_dists = jnp.asarray(
        jnp.where(min_dist_mask, dists_squared, jnp.inf)
    )
    min_cubed_dists = jnp.min(masked_dists, axis=1)
    shape_loss = jnp.sum(min_cubed_dists * boundary_mask)

    return shape_loss


def _loss_fn(
        logits, knot_ctx, goal_area_bounds, min_dist_mask, n_growth_steps,
        jax_arrays, params
    ):
    ar_logits, el_logits = logits[:2]

    knot_ctx = _update_knot_ctx(logits, knot_ctx, params.knots)

    goal_areas = _calc_goal_areas(
        goal_area_bounds, ar_logits, jax_arrays['proximal_mask'], params.knots,
        knot_ctx
    )
    goal_elongations = _calc_goal_elongations(
        el_logits, jax_arrays['proximal_mask'], params.knots, knot_ctx
    )

    growth_evolution = morphing.iterate(
        goal_areas, goal_elongations, n_growth_steps, jax_arrays, params
    )
    final_vertices = growth_evolution[-1]

    loss = params.shape_loss_weight * _calc_shape_loss(
        final_vertices, jax_arrays['boundary_mask'], jax_arrays['outer_shape'],
        min_dist_mask
    )
    aux_data = (final_vertices, knot_ctx)

    return loss, aux_data


loss_fn = jax.jit(
    jax.value_and_grad(_loss_fn, has_aux=True, argnums=0),
    static_argnames=['n_growth_steps']
)


def _calc_knots_to_tutte_centroids_dist_vecs(knots, jax_arrays):
    dist_vecs = jax_arrays['tutte_centroids'][:, None] - knots[None, :]
    return dist_vecs


def _find_closest_polygon_by_knots(knots, jax_arrays):
    dist_vecs = _calc_knots_to_tutte_centroids_dist_vecs(knots, jax_arrays)
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


def _calc_logits(areas, elongations, goal_area_bounds):
    min_area, max_area = goal_area_bounds
    areas = areas.clip(min_area + 1e-8, max_area - 1e-8)
    ar_logits = _calc_inverse_areas(min_area, max_area, areas)
    el_logits = _calc_inverse_elongations(elongations)
    return ar_logits, el_logits


def _get_poly_init_logits(tutte_areas, tutte_elongations, goal_area_bounds):
    ar_logits, el_logits = _calc_logits(
        tutte_areas, tutte_elongations, goal_area_bounds
    )
    init_logits = (ar_logits, el_logits)
    return init_logits


def _get_knot_init_logits(
        jax_arrays, tutte_areas, tutte_elongations, goal_area_bounds
    ):
    knot_positions = ['left', 'center']
    left_and_center_ar_logits = []
    left_and_center_el_logits = []
    for pos in knot_positions:
        knots = jax_arrays[pos + '_knots']
        closest_polygon_by_knots = _find_closest_polygon_by_knots(
            knots, jax_arrays
        )

        areas = _calc_mean_closest_metric(
            tutte_areas, closest_polygon_by_knots, jax_arrays
        )
        elongations = _calc_mean_closest_metric(
            tutte_elongations, closest_polygon_by_knots, jax_arrays
        )
        ar_logits, el_logits = _calc_logits(
            areas, elongations, goal_area_bounds
        )
        left_and_center_ar_logits.append(ar_logits)
        left_and_center_el_logits.append(el_logits)

    ar_logits = jnp.concatenate(left_and_center_ar_logits)
    el_logits = jnp.concatenate(left_and_center_el_logits)
    std_logits = _calc_std_logits(jax_arrays)

    init_logits = (ar_logits, el_logits, std_logits)
    return init_logits


def _get_init_logits(goal_area_bounds, jax_arrays, params):
    if params.knots:
        init_logits = _get_knot_init_logits(
            jax_arrays, jax_arrays['tutte_areas'],
            jax_arrays['tutte_elongations'], goal_area_bounds
        )
    else:
        init_logits = _get_poly_init_logits(
            jax_arrays['tutte_areas'], jax_arrays['tutte_elongations'],
            goal_area_bounds
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


@dataclass
class _BestState:
    loss: float
    final_vertices: jnp.ndarray
    goal_areas: jnp.ndarray
    goal_elongations: jnp.ndarray
    final_areas: jnp.ndarray
    final_elongations: jnp.ndarray


def _assemble_tabular_output(best):
    tabular_output = {
        'best_goal_area': best.goal_areas,
        'best_goal_elongation': best.goal_elongations,
        'final_area': best.final_areas,
        'final_elongation': best.final_elongations,
    }
    return tabular_output


def _iterate_towards_shape(logits, goal_area_bounds, jax_arrays, params):
    vertices = jax_arrays['init_vertices']

    min_dist_mask = _make_min_dist_mask(jax_arrays)

    knot_ctx = _get_knot_ctx(params.knots, jax_arrays)

    optimizer = _MyOptimizer(logits)

    best = _BestState(
        loss = jnp.inf,
        final_vertices = jnp.array([]),
        goal_areas = jnp.array([]),
        goal_elongations = jnp.array([]),
        final_areas = jnp.array([]),
        final_elongations = jnp.array([])
    )

    final_tissues = [vertices]

    steps_since_best_loss = 0

    for shape_step in range(params.n_shape_steps):
        (loss, aux_data), grads = (
            loss_fn(
                logits, knot_ctx, goal_area_bounds, min_dist_mask,
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

            best.final_vertices = vertices
            best.goal_areas = _calc_goal_areas(
                goal_area_bounds, ar_logits, jax_arrays['proximal_mask'],
                params.knots, knot_ctx
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

    tabular_output = _assemble_tabular_output(best)

    return best.loss, final_tissues, best, tabular_output


def run(params):
    jax_arrays = my_utils.get_jax_arrays(params)

    goal_area_bounds = _calc_goal_area_bounds(
        jax_arrays['tutte_areas'], params
    )

    init_logits = _get_init_logits(goal_area_bounds, jax_arrays, params)

    best_loss, final_tissues, best, tabular_output = _iterate_towards_shape(
        init_logits, goal_area_bounds, jax_arrays, params
    )

    return best_loss, final_tissues, best, tabular_output
