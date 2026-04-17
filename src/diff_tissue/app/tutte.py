import matplotlib.pyplot as plt
import numpy as np

from ..core import init_systems, metrics
from . import io_utils, parameters


class _TuttePaths:
    def __init__(self, project_paths):
        self._project_paths = project_paths

    @property
    def output_dir(self):
        output_dir_ = self._project_paths.make_subdir(
            self._project_paths.outputs_base_dir, "tutte"
        )
        return output_dir_


def _add_artists(ax, indices, valid_mask, vertices):
    for i in range(indices.shape[0]):
        vertex_inds = indices[i][valid_mask[i]]
        polygon = vertices[vertex_inds]
        ax.scatter(
            polygon[:, 0], polygon[:, 1], s=2.0, color="green", zorder=1
        )
        ax.plot(polygon[:, 0], polygon[:, 1], lw=0.7, color="black", zorder=2)


def _plot_mapping(
    init_vertices, indices, valid_mask, tutte_vertices, output_path
):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Initial mesh
    ax = axs[0]
    _add_artists(ax, indices, valid_mask, init_vertices)
    ax.set_aspect("equal")
    ax.set_title("Initial mesh")

    # Tutte mesh
    ax = axs[1]
    _add_artists(ax, indices, valid_mask, tutte_vertices)
    ax.set_aspect("equal")
    ax.set_title("Tutte mesh")

    # Vector field from initial to Tutte
    ax = axs[2]
    ax.quiver(
        init_vertices[:, 0],
        init_vertices[:, 1],
        tutte_vertices[:, 0] - init_vertices[:, 0],
        tutte_vertices[:, 1] - init_vertices[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="r",
    )

    offset = 3.5
    minvals = tutte_vertices.min(axis=0)
    maxvals = tutte_vertices.max(axis=0)
    xlim = np.array([minvals[0] - offset, maxvals[0] + offset])
    ylim = np.array([minvals[1] - offset, maxvals[1] + offset])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title("Vector Field: Initial → Tutte")

    fig.tight_layout()
    io_utils.save_pdf(output_path, fig)


def plot(params, paths):
    polygons = init_systems.get_system(params)
    tutte_metrics = metrics.get_tutte_metrics(params)

    tutte_paths = _TuttePaths(paths)
    param_string = parameters.get_param_string(params)
    output_path = tutte_paths.output_dir / f"{param_string}.pdf"

    _plot_mapping(
        polygons.init_vertices,
        polygons.indices,
        polygons.valid_mask,
        tutte_metrics.vertices,
        output_path,
    )
