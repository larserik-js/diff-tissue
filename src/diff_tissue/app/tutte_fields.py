from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from . import io_utils, parameters
from ..core import tutte_fields as tutte_fields_core
from ..core import init_systems, shapes


OUTPUT_TYPE_DIR = "tutte_fields"


def _get_general_target_boundary(shape):
    general_params = parameters.Params(system="few", shape=shape, seed=0)
    polygons = init_systems.get_system(general_params)
    target_boundary = shapes.get_target_boundary(general_params, polygons)
    return target_boundary.vertices


def _get_meshes(output_manager, shape):
    meshes_file = output_manager.cache_path(f"meshes__{shape}.pkl")
    if meshes_file.exists():
        meshes = io_utils.load_pkl(meshes_file)
    else:
        params = parameters.Params(shape=shape)
        meshes = tutte_fields_core.build_meshes(params)
        io_utils.save_pkl(meshes_file, meshes)
    return meshes


def _generate_fields(output_manager, shape):
    target_boundary = _get_general_target_boundary(shape)
    points_inside_shape = tutte_fields_core.get_points_inside_shape(
        target_boundary, nx=100, ny=100
    )

    meshes = _get_meshes(output_manager, shape)

    area_field, anisotropy_field = tutte_fields_core.get_fields(
        meshes, points_inside_shape
    )

    tutte_fields_ = tutte_fields_core.TutteFields(
        points_inside_shape, area_field, anisotropy_field
    )
    return tutte_fields_


def get_fields(shape, output):
    tutte_fields_file = output.cache_path(f"fields__{shape}.pkl")
    if tutte_fields_file.exists():
        tutte_fields_ = io_utils.load_pkl(tutte_fields_file)
    else:
        tutte_fields_ = _generate_fields(output, shape)
        io_utils.save_pkl(tutte_fields_file, tutte_fields_)
    return tutte_fields_


def _add_colorbar(ax, cmap_vals, cmap_name):
    normalize = colors.Normalize(
        vmin=np.nanmin(cmap_vals), vmax=np.nanmax(cmap_vals)
    )
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array(cmap_vals)
    ax.figure.colorbar(sm, ax=ax, shrink=0.8)


def plot(tutte_fields):
    coords = tutte_fields.coords
    tutte_fields = [tutte_fields.areas, tutte_fields.anisotropies]

    titles = ["Tutte areas", "Tutte anisotropies"]
    cmaps = ["copper", "viridis"]

    fig, axs = plt.subplots(2, figsize=(5, 6))
    for i, ax in enumerate(axs):
        tutte_field = tutte_fields[i]
        ax.scatter(
            coords[:, 0], coords[:, 1], c=tutte_field, cmap=cmaps[i], s=1.5
        )
        _add_colorbar(ax, tutte_field, cmaps[i])
        ax.set_title(titles[i])
        ax.set_aspect("equal")
        ax.set_xlim(-11.0, 11.0)
        ax.set_ylim(-0.5, 16.1)
        if i == 0:
            ax.set_xticklabels([])
        if i == 1:
            ax.set_xlabel("$x$")

    fig.tight_layout()
    return fig


def save_plot(fig, shape, output):
    output_file = output.file_path(f"tutte_fields__{shape}.pdf")
    fig.savefig(output_file)
