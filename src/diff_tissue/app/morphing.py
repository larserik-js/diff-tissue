import numpy as np

from ..core.jax_bootstrap import jnp
from ..core import metrics
from ..core import morphing as morphing_core
from . import io_utils, plotting


class MorphingPaths:
    def __init__(self, project_paths, param_string):
        self._project_paths = project_paths
        self._param_string = param_string
        self._output_type_dir_name = "morphing"

    @property
    def _data_dir(self):
        data_dir_ = self._project_paths.make_dir(
            self._project_paths.processed_data_dir, self._output_type_dir_name
        )
        return data_dir_

    @property
    def data_path(self):
        data_path = self._data_dir / f"{self._param_string}.npz"
        return data_path

    @property
    def output_dir(self):
        output_dir_ = self._project_paths.make_dir(
            self._project_paths.outputs_base_dir,
            self._output_type_dir_name,
            self._param_string,
        )
        return output_dir_


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

    return np.array(morph_evolution)


def get_morph_evolution(polygons, params, data_path):
    if data_path.exists():
        data = io_utils.load_dict_of_arrays(data_path)
        morph_evolution = data["morph_evolution"]
    else:
        morph_evolution = _morph(polygons, params)
        io_utils.save_arrays(data_path, morph_evolution=morph_evolution)
    return morph_evolution
