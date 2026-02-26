from ..core.jax_bootstrap import jax, jnp
from ..core import morphing as morphing_core
from ..core import my_utils
from . import io_utils, plotting


OUTPUT_TYPE_DIR = "morphing"


def save_figs(growth_evolution, output, param_string, params):
    figure = plotting.MorphFigure(params)

    for t, vertices in enumerate(growth_evolution):
        if t % 10 == 0:
            fig_path = output.file_path(param_string, f"step={t:03d}.png")
            figure.save_plot(vertices, fig_path)
    fig_path = output.file_path(param_string, f"step={t:03d}.png")
    figure.save_plot(vertices, fig_path)


jiterate = jax.jit(morphing_core.iterate, static_argnames=["n_steps"])


def _morph(jax_arrays, params):
    poly_metrics = my_utils.initialize_poly_metrics(
        vertices=jax_arrays["init_vertices"],
        indices=jax_arrays["indices"],
        valid_mask=jax_arrays["valid_mask"],
    )
    init_areas = poly_metrics.areas

    goal_areas = 2.0 * init_areas
    goal_anisotropies = 5.0 * jnp.ones_like(init_areas)

    growth_evolution = jiterate(
        goal_areas,
        goal_anisotropies,
        params.n_growth_steps,
        jax_arrays,
        params,
    )

    return growth_evolution


def get_growth_evolution(cache_path, jax_arrays, params):
    if cache_path.exists():
        growth_evolution = io_utils.load_pkl(cache_path)
    else:
        growth_evolution = _morph(jax_arrays, params)
        io_utils.save_pkl(cache_path, growth_evolution)
    return growth_evolution
