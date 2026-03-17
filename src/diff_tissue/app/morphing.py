from ..core.jax_bootstrap import jax, jnp
from ..core import metrics
from ..core import morphing as morphing_core
from . import io_utils, plotting


OUTPUT_TYPE_DIR = "morphing"


def save_figs(morph_evolution, output, param_string, params):
    figure = plotting.MorphFigure(params)

    for t, vertices in enumerate(morph_evolution):
        if t % 10 == 0:
            fig_path = output.file_path(param_string, f"step={t:03d}.png")
            figure.save_plot(vertices, fig_path)
    fig_path = output.file_path(param_string, f"step={t:03d}.png")
    figure.save_plot(vertices, fig_path)


jiterate = jax.jit(morphing_core.iterate, static_argnames=["n_steps"])


def _morph(polygons, params):
    poly_metrics = metrics.initialize_poly_metrics(
        vertices=polygons.init_vertices,
        indices=polygons.indices,
        valid_mask=polygons.valid_mask,
    )
    init_areas = poly_metrics.areas

    goal_areas = 2.0 * init_areas
    goal_anisotropies = 5.0 * jnp.ones_like(init_areas)

    morph_evolution = jiterate(
        goal_areas,
        goal_anisotropies,
        params.n_morph_steps,
        polygons,
        params,
    )

    return morph_evolution


def get_morph_evolution(cache_path, polygons, params):
    if cache_path.exists():
        morph_evolution = io_utils.load_pkl(cache_path)
    else:
        morph_evolution = _morph(polygons, params)
        io_utils.save_pkl(cache_path, morph_evolution)
    return morph_evolution
