from dataclasses import dataclass, field
import optax
from typing import cast

from diff_tissue.app import parameters

from .jax_bootstrap import jax, jnp, struct
from . import init_systems, morphing, my_utils, poly_identities


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
    min_goal_area = 0.0
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


def _calc_goal_areas(goal_area_bounds, ar_logits, knots, knot_ctx):
    goal_areas = _calc_goal_areas_(goal_area_bounds, ar_logits)

    if knots:
        goal_areas = _knots_to_full_shape(
            goal_areas, knot_ctx.n_left_logits, knot_ctx.knot_weights
        )

    return goal_areas


def _calc_goal_anisotropies_(an_logits):
    goal_anisotropies = _calc_sigmoid(-1.0, 1.0, an_logits)
    return goal_anisotropies


def _calc_goal_anisotropies(an_logits, knots, knot_ctx):
    goal_anisotropies = _calc_goal_anisotropies_(an_logits)

    if knots:
        goal_anisotropies = _knots_to_full_shape(
            goal_anisotropies, knot_ctx.n_left_logits, knot_ctx.knot_weights
        )

    return goal_anisotropies


def _make_min_dist_mask(jax_arrays):
    min_dist_mask = jnp.ones(
        (
            jax_arrays["boundary_inds"].shape[0],
            jax_arrays["target_boundary"].shape[0],
        ),
        dtype=bool,
    )
    fixed_boundary_mask = ~jax_arrays["free_mask"][jax_arrays["boundary_inds"]]
    fixed_mask = jnp.any(fixed_boundary_mask, axis=1)

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


def _get_knot_ctx(use_knots, knots, tutte_centroids):
    if use_knots:
        dist_vecs = _calc_knots_to_tutte_centroids_dist_vecs(
            knots.all_knots, tutte_centroids
        )
        knot_ctx = _KnotCtx(
            n_left_logits=knots.left_knots.shape[0],
            dist_vecs=dist_vecs,
            knot_weights=jnp.array([]),
        )
    else:
        knot_ctx = None

    return knot_ctx


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


def _update_knot_ctx(logits, knot_ctx, knots):
    if knots:
        updated_knot_weights = _calc_knot_weights(
            logits.std_logits, knot_ctx.dist_vecs
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


def _get_segments(vertices):
    closed_polygon = jnp.concatenate([vertices, vertices[:1]], axis=0)
    segments = closed_polygon[1:] - closed_polygon[:-1]
    return segments


def _calc_mesh_target_loss(
    first_boundary_vertices,
    second_boundary_vertices,
    second_boundary_segments,
    min_dist_mask,
):
    dists_squared = _calc_dists_squared(
        second_boundary_vertices,
        second_boundary_segments,
        first_boundary_vertices,
    )
    masked_dists = jnp.asarray(
        jnp.where(min_dist_mask, dists_squared, jnp.inf)
    )
    min_squared_dists = jnp.min(masked_dists, axis=1)
    target_to_mesh_loss = jnp.mean(min_squared_dists)

    return target_to_mesh_loss


def _calc_shape_loss(
    boundary_vertices,
    target_boundary,
    target_boundary_segments,
    min_dist_mask,
):
    mesh_to_target_loss = _calc_mesh_target_loss(
        boundary_vertices,
        target_boundary,
        target_boundary_segments,
        min_dist_mask,
    )

    boundary_segments = _get_segments(boundary_vertices)
    min_dist_mask = min_dist_mask.T

    target_to_mesh_loss = _calc_mesh_target_loss(
        target_boundary, boundary_vertices, boundary_segments, min_dist_mask
    )

    shape_loss = mesh_to_target_loss + target_to_mesh_loss

    return shape_loss


def _calc_area_id_loss(poly_metrics, poly_ids):
    proximal_areas = poly_metrics.areas[poly_ids.proximal_inds]
    distal_areas = poly_metrics.areas[poly_ids.distal_inds]

    proximal_to_distal_scale = 2.0

    target_area = proximal_to_distal_scale * jnp.mean(distal_areas)
    proximal_loss = jnp.mean(jnp.square(proximal_areas - target_area))

    target_area = jnp.mean(proximal_areas) / proximal_to_distal_scale
    distal_loss = jnp.mean(jnp.square(distal_areas - target_area))

    area_loss = proximal_loss + distal_loss

    return area_loss


def _calc_anisotropy_id_loss(poly_metrics, poly_ids):
    proximal_anisotropies = poly_metrics.anisotropies[poly_ids.proximal_inds]
    distal_anisotropies = poly_metrics.anisotropies[poly_ids.distal_inds]

    # Interpolation between the distal mean and the right limit (1.0)
    t = 1.0
    target_anisotropy = (1.0 - t) * jnp.mean(distal_anisotropies) + t * 1.0

    proximal_loss = jnp.mean(
        jnp.square(proximal_anisotropies - target_anisotropy)
    )

    # Interpolation between the proximal mean and 0.0
    t = 1.0
    target_anisotropy = (1.0 - t) * jnp.mean(proximal_anisotropies)
    distal_loss = jnp.mean(jnp.square(distal_anisotropies - target_anisotropy))

    anisotropy_loss = proximal_loss + distal_loss

    return anisotropy_loss


def _calc_poly_id_loss(poly_ids, poly_metrics):
    if poly_ids is None:
        poly_id_loss = 0.0
    else:
        area_loss = _calc_area_id_loss(poly_metrics, poly_ids)
        anisotropy_loss = _calc_anisotropy_id_loss(poly_metrics, poly_ids)

        poly_id_loss = area_loss + anisotropy_loss

    return poly_id_loss


def _loss_fn(
    logits,
    knot_ctx,
    goal_area_bounds,
    min_dist_mask,
    n_growth_steps,
    poly_metrics,
    poly_ids,
    jax_arrays,
    params,
):
    knot_ctx = _update_knot_ctx(logits, knot_ctx, params.knots)

    goal_areas = _calc_goal_areas(
        goal_area_bounds,
        logits.ar_logits,
        params.knots,
        knot_ctx,
    )
    goal_anisotropies = _calc_goal_anisotropies(
        logits.an_logits, params.knots, knot_ctx
    )

    growth_evolution = morphing.iterate(
        goal_areas, goal_anisotropies, n_growth_steps, jax_arrays, params
    )
    final_vertices = growth_evolution[-1]

    boundary_vertices = final_vertices[jax_arrays["boundary_inds"]]

    poly_metrics = my_utils.update_poly_metrics(poly_metrics, final_vertices)

    shape_loss = params.shape_loss_weight * _calc_shape_loss(
        boundary_vertices,
        jax_arrays["target_boundary"],
        jax_arrays["target_boundary_segments"],
        min_dist_mask,
    )

    poly_id_loss = params.poly_id_loss_weight * _calc_poly_id_loss(
        poly_ids, poly_metrics
    )

    loss = shape_loss + poly_id_loss

    aux_data = (
        final_vertices,
        goal_areas,
        goal_anisotropies,
        knot_ctx,
        poly_metrics,
    )

    return loss, aux_data


loss_fn = jax.jit(
    jax.value_and_grad(_loss_fn, has_aux=True, argnums=0),
    static_argnames=["n_growth_steps"],
)


@struct.dataclass
class _Logits:
    ar_logits: jnp.ndarray
    an_logits: jnp.ndarray
    std_logits: jnp.ndarray | None = None


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


def _calc_std_logits(knots):
    knots_x_diff = knots.center_knots[0, 0] - knots.left_knots[-1, 0]
    knots_y_diff = knots.left_knots[-1, 1] - knots.left_knots[-2, 1]

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
    init_logits = _Logits(ar_logits=ar_logits, an_logits=an_logits)
    return init_logits


def _get_knot_init_logits(
    knots,
    tutte_centroids,
    tutte_areas,
    tutte_anisotropies,
    goal_area_bounds,
):
    knot_positions = [knots.left_knots, knots.center_knots]
    left_and_center_ar_logits = []
    left_and_center_an_logits = []
    for knots_subset in knot_positions:
        closest_polygon_by_knots = _find_closest_polygon_by_knots(
            knots_subset, tutte_centroids
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
    std_logits = _calc_std_logits(knots)

    init_logits = _Logits(
        ar_logits=ar_logits, an_logits=an_logits, std_logits=std_logits
    )
    return init_logits


def _get_init_logits(goal_area_bounds, knots, tutte_metrics, params):
    if params.knots:
        init_logits = _get_knot_init_logits(
            knots,
            tutte_metrics.centroids,
            tutte_metrics.areas,
            tutte_metrics.anisotropies,
            goal_area_bounds,
        )
    else:
        init_logits = _get_poly_init_logits(
            tutte_metrics.areas,
            tutte_metrics.anisotropies,
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

    def update(self, logits, grads) -> _Logits:
        updates, self._state = self._optimizer.update(grads, self._state)
        updated_logits = optax.apply_updates(logits, updates)
        updated_logits = cast(_Logits, updated_logits)
        return updated_logits


@dataclass
class _SimStates:
    loss_vals: list[float] = field(default_factory=list)
    valid: list[bool] = field(default_factory=list)
    final_vertices: list[jnp.ndarray] = field(default_factory=list)
    goal_areas: list[jnp.ndarray] = field(default_factory=list)
    goal_anisotropies: list[jnp.ndarray] = field(default_factory=list)
    final_areas: list[jnp.ndarray] = field(default_factory=list)
    final_anisotropies: list[jnp.ndarray] = field(default_factory=list)


@dataclass
class BestState:
    loss: float
    final_vertices: jnp.ndarray
    goal_areas: jnp.ndarray
    goal_anisotropies: jnp.ndarray
    final_areas: jnp.ndarray
    final_anisotropies: jnp.ndarray


def _get_valid_best_idx(sim_state):
    masked_loss_vals = jnp.asarray(
        jnp.where(
            jnp.array(sim_state.valid), jnp.array(sim_state.loss_vals), jnp.inf
        )
    )
    best_index = jnp.argmin(masked_loss_vals)
    return best_index


def get_best_state(sim_state):
    best_index = _get_valid_best_idx(sim_state)
    best = BestState(
        loss=sim_state.loss_vals[best_index],
        final_vertices=sim_state.final_vertices[best_index],
        goal_areas=sim_state.goal_areas[best_index],
        goal_anisotropies=sim_state.goal_anisotropies[best_index],
        final_areas=sim_state.final_areas[best_index],
        final_anisotropies=sim_state.final_anisotropies[best_index],
    )
    return best


def _validate(final_areas):
    all_areas_positive = bool(~jnp.any(final_areas < 0.0))
    return all_areas_positive


def _iterate_towards_shape(
    logits: _Logits,
    knot_ctx: _KnotCtx | None,
    goal_area_bounds: tuple,
    jax_arrays: dict,
    params: parameters.Params,
) -> _SimStates:
    vertices = jax_arrays["init_vertices"]

    min_dist_mask = _make_min_dist_mask(jax_arrays)

    optimizer = _MyOptimizer(logits)

    poly_metrics = my_utils.initialize_poly_metrics(
        vertices=vertices,
        indices=jax_arrays["indices"],
        valid_mask=jax_arrays["valid_mask"],
    )

    poly_ids = poly_identities.get_poly_identities(params)

    sim_states = _SimStates()

    best_loss = jnp.inf
    steps_since_best_loss = 0

    for shape_step in range(params.n_shape_steps):
        (loss, aux_data), grads = loss_fn(
            logits,
            knot_ctx,
            goal_area_bounds,
            min_dist_mask,
            params.n_growth_steps,
            poly_metrics,
            poly_ids,
            jax_arrays,
            params,
        )
        vertices, goal_areas, goal_anisotropies, knot_ctx, poly_metrics = (
            aux_data
        )

        poly_metrics = my_utils.update_poly_metrics(poly_metrics, vertices)

        sim_states.loss_vals.append(loss)
        sim_states.final_vertices.append(vertices)
        sim_states.goal_areas.append(goal_areas)
        sim_states.goal_anisotropies.append(goal_anisotropies)
        sim_states.final_areas.append(poly_metrics.areas)
        sim_states.final_anisotropies.append(poly_metrics.anisotropies)

        if not params.quiet:
            print(f"{shape_step}: Shape loss = {loss}")

        valid_sim = _validate(poly_metrics.areas)
        sim_states.valid.append(valid_sim)

        if loss < best_loss and valid_sim:
            best_loss = loss
            steps_since_best_loss = 0

            if not params.quiet:
                print("(New best loss.)")
                print("")
        else:
            steps_since_best_loss += 1

        if steps_since_best_loss >= 50 and best_loss != jnp.inf:
            if not params.quiet:
                print("(Stopped - iteration diverged.)")
                print("")
            break
        else:
            logits = optimizer.update(logits, grads)

    return sim_states


def run(params):
    jax_arrays = my_utils.get_jax_arrays(params)

    tutte_metrics = my_utils.get_tutte_metrics(params)

    knots = init_systems.Knots()

    knot_ctx = _get_knot_ctx(params.knots, knots, tutte_metrics.centroids)

    goal_area_bounds = _calc_goal_area_bounds(tutte_metrics.areas, params)

    init_logits = _get_init_logits(
        goal_area_bounds, knots, tutte_metrics, params
    )

    sim_states = _iterate_towards_shape(
        init_logits, knot_ctx, goal_area_bounds, jax_arrays, params
    )
    return sim_states
