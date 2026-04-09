from ..core.jax_bootstrap import jnp
from ..core import metrics
from ..core import morphing as morphing_core
from . import io_utils, plotting


OUTPUT_TYPE_DIR = "morphing"


def save_figs(morph_evolution, params, output_dir):
    figure = plotting.MorphFigure(params)

    for t, vertices in enumerate(morph_evolution):
        if t % 10 == 0 or t == len(morph_evolution) - 1:
            figure.update(vertices)
            fig_path = output_dir / f"step={t:03d}.png"
            io_utils.save_pdf(fig_path, figure.fig, dpi=100)


def _morph(polygons, params):
    poly_metrics = metrics.initialize_poly_metrics(
        vertices=polygons.init_vertices,
        indices=polygons.indices,
        valid_mask=polygons.valid_mask,
    )
    init_areas = poly_metrics.areas

    goal_areas = 2.0 * init_areas
    goal_anisotropies = 5.0 * jnp.ones_like(init_areas)

    morph_evolution = morphing_core.iterate(
        goal_areas,
        goal_anisotropies,
        params.n_morph_steps,
        polygons,
        params,
    )

    return morph_evolution


def get_morph_evolution(polygons, params, data_path):
    if data_path.exists():
        morph_evolution = io_utils.load_arrays(data_path)
    else:
        morph_evolution = _morph(polygons, params)
        io_utils.save_arrays(data_path, morph_evolution)
    return morph_evolution
