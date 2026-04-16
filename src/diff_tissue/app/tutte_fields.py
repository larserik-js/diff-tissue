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


def _get_meshes(shape, data_path):
    if data_path.exists():
        meshes = io_utils.load_pkl(data_path)
    else:
        params = parameters.Params(shape=shape)
        meshes = tutte_fields_core.build_meshes(params)
        io_utils.save_pkl(data_path, meshes)
    return meshes


def _load_tutte_fields(path):
    data_dict = io_utils.load_dict_of_arrays(path)
    return tutte_fields_core.TutteFields(**data_dict)


def _generate_fields(shape, meshes_data_path):
    target_boundary = _get_general_target_boundary(shape)
    points_inside_shape = tutte_fields_core.get_points_inside_shape(
        target_boundary, nx=100, ny=100
    )

    meshes = _get_meshes(shape, meshes_data_path)

    area_field, anisotropy_field = tutte_fields_core.get_fields(
        meshes, points_inside_shape
    )

    tutte_fields_ = tutte_fields_core.TutteFields(
        points_inside_shape, area_field, anisotropy_field
    )
    return tutte_fields_


def get_fields(shape, paths):
    data_dir = paths.make_subdir(paths.processed_data_dir, OUTPUT_TYPE_DIR)
    data_path = data_dir / f"fields__{shape}.npz"
    if data_path.exists():
        tutte_fields_ = _load_tutte_fields(data_path)
    else:
        meshes_data_dir = paths.make_subdir(
            paths.interim_data_dir, OUTPUT_TYPE_DIR
        )
        meshes_data_path = meshes_data_dir / f"meshes__{shape}.pkl"
        tutte_fields_ = _generate_fields(shape, meshes_data_path)
        io_utils.save_arrays_from_dataclass(data_path, tutte_fields_)
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


def save_plot(fig, shape, output_dir):
    output_file = output_dir / f"tutte_fields__{shape}.pdf"
    io_utils.save_pdf(output_file, fig)
