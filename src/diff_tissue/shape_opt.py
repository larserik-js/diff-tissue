from dataclasses import dataclass

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

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
    min_area_scaling = 1 / params['growth_scale']
    goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params['max_area_scaling'], ar_logits,
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

    min_area_scaling = 1 / params['growth_scale']
    goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, params['max_area_scaling'], ar_logits,
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


def _find_closest_vertex_inds_by_knots(knots, mapped_vertices):
    """
    Find the closest mapped vertex for each knot.

    Args:
        knots (jnp.ndarray): First array of shape (M,2).
        mapped_vertices (jnp.ndarray): Second array of shape (N,2).

    Returns:
        jnp.ndarray: Array of closest mapped vertex for each knot, shape (M,).
    """
    dist_vecs = knots[:,None,:] - mapped_vertices
    dists = jnp.linalg.norm(dist_vecs, axis=2)
    closest_vertices = jnp.argmin(dists, axis=1)
    return closest_vertices


def _find_closest_polygons_by_knots(knots, mapped_vertices, vertex_polygons):
    """
    Find the closest polygons for each knot.

    First find the closest mapped vertex for each knot. For each of those
    vertices, choose the polygons which belong to it.

    Args:
        knots (jnp.ndarray): First array of shape (M,2).
        mapped_vertices (jnp.ndarray): Second array of shape (N,2).
        vertex_polygons (jnp.ndarray): Third array of shape (N,3).

    Returns:
        jnp.ndarray: Array of closest polygons for each knot, shape (M,3).
        The array is padded with -1 where the vertices belong to fewer than
        3 polygons.
    """
    closest_vertex_inds_by_knots = _find_closest_vertex_inds_by_knots(
        knots, mapped_vertices
    )
    closest_polygons_by_knots = vertex_polygons[closest_vertex_inds_by_knots]
    return closest_polygons_by_knots


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


def _calc_logits(params, area_scalings, elongations):
    min_area_scaling = 1 / params.numerical['growth_scale']
    max_area_scaling = params.numerical['max_area_scaling']
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


def _get_knot_init_logits(jax_arrays, params, mapped_vertices, mapped_areas,
                          mapped_elongations, init_areas):
    knot_positions = ['left', 'center']
    left_and_center_ar_logits = []
    left_and_center_el_logits = []
    for pos in knot_positions:
        knots = jax_arrays[pos + '_knots']
        closest_polygons_by_knots = _find_closest_polygons_by_knots(
            knots, mapped_vertices, jax_arrays['vertex_polygons']
        )
        area_scalings = _get_knots_area_scalings(
            closest_polygons_by_knots, mapped_areas, init_areas
        )
        elongations = _calc_mean_closest_metric(
            mapped_elongations, closest_polygons_by_knots
        )
        ar_logits, el_logits = _calc_logits(
            params, area_scalings, elongations
        )
        left_and_center_ar_logits.append(ar_logits)
        left_and_center_el_logits.append(el_logits)

    ar_logits = jnp.concatenate(left_and_center_ar_logits)
    el_logits = jnp.concatenate(left_and_center_el_logits)

    init_smoothing_stds = jnp.array([5.0, 1.0])
    std_logits = _calc_inverse_smoothing_stds(init_smoothing_stds)

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
            jax_arrays, params, jax_arrays['mapped_vertices'], mapped_areas,
            mapped_elongations, init_areas
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


def _calc_dist_vecs(points, all_knots):
    dist_vecs = points[:, None] - all_knots[None, :]
    return dist_vecs


def _get_knot_ctx(knots, jax_arrays):
    if knots:
        dist_vecs = _calc_dist_vecs(
            jax_arrays['mapped_centroids'], jax_arrays['all_knots']
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


def _assemble_tabular_output(
        vertices, init_areas, min_area_scaling, logits, best, jax_arrays,
        all_params, knot_ctx
    ):
    ar_logits, el_logits = logits[:2]

    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
    final_areas = my_utils.calc_all_areas(
        all_cells, jax_arrays['valid_mask']
    )
    final_elongations = my_utils.calc_elongations(
        all_cells, jax_arrays['valid_mask']
    )
    final_goal_areas = _calc_goal_areas(
        init_areas, min_area_scaling, all_params.numerical['max_area_scaling'],
        ar_logits, jax_arrays['proximal_mask'], all_params.knots, knot_ctx
    )
    final_goal_elongations = _calc_goal_elongations(
        el_logits, jax_arrays['proximal_mask'], all_params.knots, knot_ctx
    )

    tabular_output = {
        'init_area': init_areas,
        'final_area': final_areas,
        'final_elongation': final_elongations,
        'best_goal_area': best.goal_areas,
        'best_goal_elongation': best.goal_elongations,
        'final_goal_area': final_goal_areas,
        'final_goal_elongation': final_goal_elongations
    }
    return tabular_output


def _iterate_towards_shape(logits, jax_arrays, all_params):
    params = all_params.numerical

    vertices = jax_arrays['init_vertices']
    all_cells = my_utils.get_all_cells(vertices, jax_arrays['indices'])
    init_areas = my_utils.calc_all_areas(all_cells, jax_arrays['valid_mask'])

    min_area_scaling = 1 / params['growth_scale']

    min_dist_mask = _make_min_dist_mask(jax_arrays)

    knot_ctx = _get_knot_ctx(all_params.knots, jax_arrays)

    optimizer = _MyOptimizer(logits)

    best = _BestState(
        loss=jnp.inf, goal_areas=jnp.array([]), goal_elongations=jnp.array([])
    )

    final_tissues = [vertices]

    steps_since_best_loss = 0

    for shape_step in range(params['n_shape_steps']):
        if all_params.knots:
            (loss, aux_data), grads = (
                _calc_loss_val_grads_knots(
                    *logits, knot_ctx, init_areas, min_dist_mask,
                    params['n_growth_steps'], jax_arrays, params
                )
            )
        else:
            (loss, aux_data), grads = (
                _calc_loss_val_grads(
                    *logits, knot_ctx, init_areas, min_dist_mask,
                    params['n_growth_steps'], jax_arrays, params
                )
            )
        vertices, knot_ctx = aux_data

        if not all_params.quiet:
            print(f'{shape_step}: Shape loss = {loss}')

        ar_logits, el_logits = logits[:2]

        if loss < best.loss:
            best.goal_areas = _calc_goal_areas(
                init_areas, min_area_scaling, params['max_area_scaling'],
                ar_logits, jax_arrays['proximal_mask'], all_params.knots,
                knot_ctx
            )
            best.goal_elongations = _calc_goal_elongations(
                el_logits, jax_arrays['proximal_mask'], all_params.knots,
                knot_ctx
            )

            best.loss = loss
            steps_since_best_loss = 0

            if not all_params.quiet:
                print(f'(Stored params with new best loss.)')
                print('')
        else:
            steps_since_best_loss += 1

        final_tissues.append(vertices)

        if steps_since_best_loss >= 20 and best.loss != jnp.inf:
            if not all_params.quiet:
                print(f'(Stopped - iteration diverged.)')
                print('')
            break
        else:
            logits = optimizer.update(logits, grads)

    final_tissues = jnp.array(final_tissues)

    tabular_output = _assemble_tabular_output(
        vertices, init_areas, min_area_scaling, logits, best, jax_arrays,
        all_params, knot_ctx
    )

    return best.loss, final_tissues, tabular_output


def run(params):
    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    init_logits = _get_init_logits(jax_arrays, params)

    best_loss, final_tissues, tabular_output = _iterate_towards_shape(
        init_logits, jax_arrays, params
    )

    return best_loss, final_tissues, tabular_output
