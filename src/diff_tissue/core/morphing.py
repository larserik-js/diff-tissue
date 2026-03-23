from .jax_bootstrap import jax, jaxopt, jnp, struct
from . import metrics


def _calc_areas_loss(target_areas, areas):
    areas_loss = jnp.mean(jnp.square(areas - target_areas))
    return areas_loss


def _calc_angles_loss(masked_cosines, optimal_angles):
    optimal_cosines = jnp.cos(optimal_angles)
    squared_diffs = jnp.square(masked_cosines - optimal_cosines)
    angles_loss = jnp.nanmean(squared_diffs)
    return angles_loss


def _calc_anisotropies_loss(target_anisotropies, anisotropies):
    anisotropy_diffs = target_anisotropies - anisotropies
    anisotropies_loss = jnp.mean(jnp.square(anisotropy_diffs))
    return anisotropies_loss


def _calc_morph_loss(
    vertices,
    target_areas,
    target_anisotropies,
    poly_metrics,
    potential_weights,
):
    poly_metrics = metrics.update_poly_metrics(poly_metrics, vertices)

    areas_loss = potential_weights.areas * _calc_areas_loss(
        target_areas, poly_metrics.areas
    )
    angles_loss = potential_weights.angles * _calc_angles_loss(
        poly_metrics.masked_cosines, poly_metrics.optimal_angles
    )
    anisotropies_loss = (
        potential_weights.anisotropies
        * _calc_anisotropies_loss(
            target_anisotropies, poly_metrics.anisotropies
        )
    )

    loss = areas_loss + angles_loss + anisotropies_loss

    return loss


def _update_targets(init_targets, goals, t_frac):
    targets = init_targets + (goals - init_targets) * t_frac
    return targets


def _lbfgs_solve(
    vertices,
    target_areas,
    target_anisotropies,
    poly_metrics,
    potential_weights,
):
    solver = jaxopt.LBFGS(fun=_calc_morph_loss, maxiter=50)
    result = solver.run(
        vertices,
        target_areas,
        target_anisotropies,
        poly_metrics,
        potential_weights,
    )
    updated_vertices = result.params

    return updated_vertices


def _update_vertices(
    vertices,
    t,
    n_morph_steps,
    goal_areas,
    goal_anisotropies,
    init_areas,
    init_anisotropies,
    poly_metrics,
    potential_weights,
    polygons,
):
    t_frac = t / n_morph_steps
    target_areas = _update_targets(init_areas, goal_areas, t_frac)
    target_anisotropies = _update_targets(
        init_anisotropies, goal_anisotropies, t_frac
    )

    updated_vertices = _lbfgs_solve(
        vertices,
        target_areas,
        target_anisotropies,
        poly_metrics,
        potential_weights,
    )
    updated_vertices = jnp.where(
        polygons.free_mask, updated_vertices, polygons.init_vertices
    )

    return updated_vertices


@struct.dataclass
class _PotentialWeights:
    areas: float
    angles: float
    anisotropies: float


def _get_potential_weights(params):
    match params.system:
        case "few":
            potential_weights = _PotentialWeights(
                areas=params.areas_pot_weight,
                angles=params.angles_pot_weight,
                anisotropies=params.anisotropies_pot_weight,
            )
        case "many":
            # TODO: Tune these weights.
            potential_weights = _PotentialWeights(
                areas=10.0,
                angles=20.0,
                anisotropies=10.0,
            )
        case _:
            raise NotImplementedError(
                "Potential weights not implemented for system: "
                f"{params.system}"
            )
    return potential_weights


def iterate(goal_areas, goal_anisotropies, n_steps, polygons, params):
    poly_metrics = metrics.initialize_poly_metrics(
        vertices=polygons.init_vertices,
        indices=polygons.indices,
        valid_mask=polygons.valid_mask,
    )
    init_areas = poly_metrics.areas
    init_anisotropies = poly_metrics.anisotropies

    potential_weights = _get_potential_weights(params)

    def update_step(carry, t):
        vertices, poly_metrics, goal_areas, goal_anisotropies = carry

        vertices = _update_vertices(
            vertices,
            t,
            params.n_morph_steps,
            goal_areas,
            goal_anisotropies,
            init_areas,
            init_anisotropies,
            poly_metrics,
            potential_weights,
            polygons,
        )

        carry = (vertices, poly_metrics, goal_areas, goal_anisotropies)

        return carry, vertices

    init_carry = (
        polygons.init_vertices,
        poly_metrics,
        goal_areas,
        goal_anisotropies,
    )

    _, morph_evolution = jax.lax.scan(
        update_step, init_carry, jnp.arange(n_steps)
    )

    return morph_evolution
