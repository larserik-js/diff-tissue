from dataclasses import dataclass

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


def _calc_inverse_anisotropies(anisotropies):
    logits = _calc_inverse_sigmoid(-1.0, 1.0, anisotropies)
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
    # This is necessary to accompany the anisotropy.
    goal_areas = jnp.where(proximal_mask, 2.0 * goal_areas, goal_areas)

    return goal_areas


def _calc_goal_anisotropies_(an_logits):
    goal_anisotropies = _calc_sigmoid(-1.0, 1.0, an_logits)
    return goal_anisotropies


def _calc_goal_anisotropies(an_logits, proximal_mask, knots, knot_ctx):
    goal_anisotropies = _calc_goal_anisotropies_(an_logits)

    if knots:
        goal_anisotropies = _knots_to_full_shape(
            goal_anisotropies, knot_ctx.n_left_logits, knot_ctx.knot_weights
        )

    # Multiplies the anisotropies of proximal polygons.
    # The actual constant does not carry physical meaning at this point.
    goal_anisotropies = jnp.where(
        proximal_mask, 3.0 * goal_anisotropies, goal_anisotropies
    )

    return goal_anisotropies


def _calc_knot_weights(std_logits, dist_vecs):
    smoothing_stds = _calc_smoothing_stds(std_logits)
    knot_weights = jnp.exp(
        -jnp.sum(
            dist_vecs**2 / (2 * smoothing_stds[None, None, :] ** 2), axis=2
        )
    )
    knot_weights += 1e-8
    knot_weights = knot_weights / jnp.sum(knot_weights, axis=1)[:, None]

    return knot_weights


def _make_min_dist_mask(jax_arrays):
    min_dist_mask = jnp.ones(
        (
            jax_arrays["init_vertices"].shape[0],
            jax_arrays["target_boundary"].shape[0],
        ),
        dtype=bool,
    )
    fixed_mask = jnp.any(~jax_arrays["free_mask"], axis=1)

    target_boundary_basal_mask = jnp.isclose(
        jax_arrays["target_boundary"][:, 1], init_systems.Coords.base_origin[1]
    )
    min_dist_mask = min_dist_mask.at[fixed_mask].set(
        target_boundary_basal_mask
    )
    return min_dist_mask


@struct.dataclass
class _KnotCtx:
    n_left_logits: int = struct.field(pytree_node=False)
    dist_vecs: jnp.ndarray
    knot_weights: jnp.ndarray


def _get_knot_ctx(knots, jax_arrays):
    if knots:
        dist_vecs = _calc_knots_to_tutte_centroids_dist_vecs(
            jax_arrays["all_knots"], jax_arrays["tutte_centroids"]
        )
        knot_ctx = _KnotCtx(
            n_left_logits=jax_arrays["left_knots"].shape[0],
            dist_vecs=dist_vecs,
            knot_weights=jnp.array([]),
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


def _expand_for_broadcasting(target_boundary, segments, final_vertices):
    target_boundary = target_boundary[None, :, :]  # (1, M, 2)
    segments = segments[None, :, :]  # (1, M, 2)
    final_vertices = final_vertices[:, None, :]  # (N, 1, 2)
    return target_boundary, segments, final_vertices


def _calc_dists_squared(segment_verts, segments, other_boundary_verts):
    segment_verts, segments, other_boundary_verts = _expand_for_broadcasting(
        segment_verts, segments, other_boundary_verts
    )
    dist_vecs = other_boundary_verts - segment_verts  # (N, M, 2)

    denom = jnp.sum(segments * segments, axis=2)  # (1, M)
    t = jnp.sum(dist_vecs * segments, axis=2) / denom  # (N, M)
    t = jax.nn.sigmoid(10.0 * (t - 0.5))  # Instead of clipping to [0, 1]

    projection = segment_verts + t[..., None] * segments  # (N, M, 2)
    dists = jnp.linalg.norm(
        other_boundary_verts - projection, axis=2
    )  # (N, M)
    dists_squared = dists**2
    return dists_squared


def _calc_mesh_to_target_loss(
    boundary_vertices,
    target_boundary,
    target_boundary_segments,
    min_dist_mask,
):
    dists_squared = _calc_dists_squared(
        target_boundary, target_boundary_segments, boundary_vertices
    )
    masked_dists = jnp.asarray(
        jnp.where(min_dist_mask, dists_squared, jnp.inf)
    )
    min_squared_dists = jnp.min(masked_dists, axis=1)
    mesh_to_target_loss = jnp.mean(min_squared_dists)

    return mesh_to_target_loss


def _get_segments(vertices):
    closed_polygon = jnp.concatenate([vertices, vertices[:1]], axis=0)
    segments = closed_polygon[1:] - closed_polygon[:-1]
    return segments


def _calc_target_to_mesh_loss(
    boundary_vertices, target_boundary, min_dist_mask
):
    boundary_segments = _get_segments(boundary_vertices)

    dists_squared = _calc_dists_squared(
        boundary_vertices, boundary_segments, target_boundary
    )
    masked_dists = jnp.asarray(
        jnp.where(min_dist_mask, dists_squared, jnp.inf)
    )
    min_squared_dists = jnp.min(masked_dists, axis=1)
    target_to_mesh_loss = jnp.mean(min_squared_dists)

    return target_to_mesh_loss


def _calc_shape_loss(
    final_vertices,
    boundary_inds,
    target_boundary,
    target_boundary_segments,
    min_dist_mask,
):
    boundary_vertices = final_vertices[boundary_inds]

    boundary_min_dist_mask = min_dist_mask[boundary_inds]
    mesh_to_target_loss = _calc_mesh_to_target_loss(
        boundary_vertices,
        target_boundary,
        target_boundary_segments,
        boundary_min_dist_mask,
    )

    boundary_min_dist_mask = boundary_min_dist_mask.T
    target_to_mesh_loss = _calc_target_to_mesh_loss(
        boundary_vertices, target_boundary, boundary_min_dist_mask
    )

    shape_loss = mesh_to_target_loss + target_to_mesh_loss

    return shape_loss


def _loss_fn(
    logits,
    knot_ctx,
    goal_area_bounds,
    min_dist_mask,
    n_growth_steps,
    jax_arrays,
    params,
):
    ar_logits, an_logits = logits[:2]

    knot_ctx = _update_knot_ctx(logits, knot_ctx, params.knots)

    goal_areas = _calc_goal_areas(
        goal_area_bounds,
        ar_logits,
        jax_arrays["proximal_mask"],
        params.knots,
        knot_ctx,
    )
    goal_anisotropies = _calc_goal_anisotropies(
        an_logits, jax_arrays["proximal_mask"], params.knots, knot_ctx
    )

    growth_evolution = morphing.iterate(
        goal_areas, goal_anisotropies, n_growth_steps, jax_arrays, params
    )
    final_vertices = growth_evolution[-1]

    loss = params.shape_loss_weight * _calc_shape_loss(
        final_vertices,
        jax_arrays["boundary_inds"],
        jax_arrays["target_boundary"],
        jax_arrays["target_boundary_segments"],
        min_dist_mask,
    )
    aux_data = (final_vertices, knot_ctx)

    return loss, aux_data


loss_fn = jax.jit(
    jax.value_and_grad(_loss_fn, has_aux=True, argnums=0),
    static_argnames=["n_growth_steps"],
)


def _calc_knots_to_tutte_centroids_dist_vecs(knots, tutte_centroids):
    dist_vecs = tutte_centroids[:, None] - knots[None, :]
    return dist_vecs


def _find_closest_polygon_by_knots(knots, tutte_centroids):
    dist_vecs = _calc_knots_to_tutte_centroids_dist_vecs(
        knots, tutte_centroids
    )
    dists = jnp.linalg.norm(dist_vecs, axis=2)
    closest_inds = jnp.argmin(dists, axis=0)
    return closest_inds


def _calc_std_logits(jax_arrays):
    knots_x_diff = (
        jax_arrays["center_knots"][0, 0] - jax_arrays["left_knots"][-1, 0]
    )
    knots_y_diff = (
        jax_arrays["left_knots"][-1, 1] - jax_arrays["left_knots"][-2, 1]
    )

    init_smoothing_stds = jnp.array([knots_x_diff, knots_y_diff])
    std_logits = _calc_inverse_smoothing_stds(init_smoothing_stds)
    return std_logits


def _calc_logits(areas, anisotropies, goal_area_bounds):
    min_area, max_area = goal_area_bounds
    areas = areas.clip(min_area + 1e-8, max_area - 1e-8)
    ar_logits = _calc_inverse_areas(min_area, max_area, areas)
    an_logits = _calc_inverse_anisotropies(anisotropies)
    return ar_logits, an_logits


def _get_poly_init_logits(tutte_areas, tutte_anisotropies, goal_area_bounds):
    ar_logits, an_logits = _calc_logits(
        tutte_areas, tutte_anisotropies, goal_area_bounds
    )
    init_logits = (ar_logits, an_logits)
    return init_logits


def _get_knot_init_logits(
    jax_arrays,
    tutte_centroids,
    tutte_areas,
    tutte_anisotropies,
    goal_area_bounds,
):
    knot_positions = ["left", "center"]
    left_and_center_ar_logits = []
    left_and_center_an_logits = []
    for pos in knot_positions:
        knots = jax_arrays[pos + "_knots"]
        closest_polygon_by_knots = _find_closest_polygon_by_knots(
            knots, tutte_centroids
        )
        closest_poly_areas = tutte_areas[closest_polygon_by_knots]
        closest_poly_anisotropies = tutte_anisotropies[
            closest_polygon_by_knots
        ]

        ar_logits, an_logits = _calc_logits(
            closest_poly_areas, closest_poly_anisotropies, goal_area_bounds
        )
        left_and_center_ar_logits.append(ar_logits)
        left_and_center_an_logits.append(an_logits)

    ar_logits = jnp.concatenate(left_and_center_ar_logits)
    an_logits = jnp.concatenate(left_and_center_an_logits)
    std_logits = _calc_std_logits(jax_arrays)

    init_logits = (ar_logits, an_logits, std_logits)
    return init_logits


def _get_init_logits(goal_area_bounds, jax_arrays, params):
    if params.knots:
        init_logits = _get_knot_init_logits(
            jax_arrays,
            jax_arrays["tutte_centroids"],
            jax_arrays["tutte_areas"],
            jax_arrays["tutte_anisotropies"],
            goal_area_bounds,
        )
    else:
        init_logits = _get_poly_init_logits(
            jax_arrays["tutte_areas"],
            jax_arrays["tutte_anisotropies"],
            goal_area_bounds,
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
            optax.scale(-1.0),
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
    goal_anisotropies: jnp.ndarray
    final_areas: jnp.ndarray
    final_anisotropies: jnp.ndarray


def _validate(final_areas):
    all_areas_positive = ~jnp.any(final_areas < 0.0)
    return all_areas_positive


def _assemble_tabular_output(best):
    tabular_output = {
        "best_goal_area": best.goal_areas,
        "best_goal_anisotropy": best.goal_anisotropies,
        "final_area": best.final_areas,
        "final_anisotropy": best.final_anisotropies,
    }
    return tabular_output


def _iterate_towards_shape(logits, goal_area_bounds, jax_arrays, params):
    vertices = jax_arrays["init_vertices"]

    min_dist_mask = _make_min_dist_mask(jax_arrays)

    knot_ctx = _get_knot_ctx(params.knots, jax_arrays)

    optimizer = _MyOptimizer(logits)

    poly_metrics = my_utils.PolyMetrics.create(
        vertices=vertices,
        indices=jax_arrays["indices"],
        valid_mask=jax_arrays["valid_mask"],
    )

    best = _BestState(
        loss=jnp.inf,
        final_vertices=jnp.array([]),
        goal_areas=jnp.array([]),
        goal_anisotropies=jnp.array([]),
        final_areas=jnp.array([]),
        final_anisotropies=jnp.array([]),
    )

    final_tissues = [vertices]

    steps_since_best_loss = 0

    for shape_step in range(params.n_shape_steps):
        (loss, aux_data), grads = loss_fn(
            logits,
            knot_ctx,
            goal_area_bounds,
            min_dist_mask,
            params.n_growth_steps,
            jax_arrays,
            params,
        )
        vertices, knot_ctx = aux_data

        if not params.quiet:
            print(f"{shape_step}: Shape loss = {loss}")

        ar_logits, an_logits = logits[:2]

        poly_metrics = poly_metrics.update(vertices)

        if loss < best.loss and _validate(poly_metrics.areas):
            best.loss = loss
            steps_since_best_loss = 0

            best.final_vertices = vertices
            best.goal_areas = _calc_goal_areas(
                goal_area_bounds,
                ar_logits,
                jax_arrays["proximal_mask"],
                params.knots,
                knot_ctx,
            )
            best.goal_anisotropies = _calc_goal_anisotropies(
                an_logits, jax_arrays["proximal_mask"], params.knots, knot_ctx
            )

            best.final_areas = poly_metrics.areas
            best.final_anisotropies = poly_metrics.anisotropies

            if not params.quiet:
                print("(Stored params with new best loss.)")
                print("")
        else:
            steps_since_best_loss += 1

        final_tissues.append(vertices)

        if steps_since_best_loss >= 20 and best.loss != jnp.inf:
            if not params.quiet:
                print("(Stopped - iteration diverged.)")
                print("")
            break
        else:
            logits = optimizer.update(logits, grads)

    final_tissues = jnp.array(final_tissues)

    tabular_output = _assemble_tabular_output(best)

    return best.loss, final_tissues, best, tabular_output


def run(params):
    jax_arrays = my_utils.get_jax_arrays(params)

    goal_area_bounds = _calc_goal_area_bounds(
        jax_arrays["tutte_areas"], params
    )

    init_logits = _get_init_logits(goal_area_bounds, jax_arrays, params)

    best_loss, final_tissues, best, tabular_output = _iterate_towards_shape(
        init_logits, goal_area_bounds, jax_arrays, params
    )

    return best_loss, final_tissues, best, tabular_output
