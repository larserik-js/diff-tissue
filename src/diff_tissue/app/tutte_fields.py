from pathlib import Path

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry as shapely_geo

from . import io_utils, parameters
from ..core import tutte_fields as tutte_fields_core
from ..core import init_systems, shapes


class TutteFieldsPaths:
    def __init__(self, project_paths):
        self._project_paths = project_paths
        self._output_type_dir = "tutte_fields"

    @property
    def fields_dir(self):
        data_dir = Path(
            self._project_paths.processed_data_dir, self._output_type_dir
        )
        io_utils.ensure_dir(data_dir)
        return data_dir

    def fields_path(self, shape):
        data_path = Path(self.fields_dir, f"fields__{shape}.npz")
        io_utils.ensure_parent_dir(data_path)
        return data_path

    @property
    def meshes_dir(self):
        data_dir = Path(
            self._project_paths.interim_data_dir, self._output_type_dir
        )
        io_utils.ensure_dir(data_dir)
        return data_dir

    def mesh_subdir(self, idx):
        mesh_subdir_ = Path(self.meshes_dir, f"mesh_{idx:03d}")
        io_utils.ensure_dir(mesh_subdir_)
        return mesh_subdir_

    @property
    def mesh_subdirs(self):
        return list(self.meshes_dir.glob("mesh*"))

    @property
    def output_dir(self):
        output_dir = Path(
            self._project_paths.outputs_base_dir, self._output_type_dir
        )
        io_utils.ensure_dir(output_dir)
        return output_dir


def _get_general_target_boundary(shape):
    general_params = parameters.Params(system="few", shape=shape, seed=0)
    polygons = init_systems.get_system(general_params)
    target_boundary = shapes.get_target_boundary(general_params, polygons)
    return target_boundary.vertices


def _save_mesh(mesh_dir, mesh):
    polygons_geo = [shapely_geo.mapping(poly) for poly in mesh.polygons]
    io_utils.save_json(mesh_dir / "geometry.json", polygons_geo)

    io_utils.save_arrays(
        mesh_dir / "arrays.npz",
        areas=mesh.areas,
        anisotropies=mesh.anisotropies,
    )


def _save_meshes(paths, meshes):
    assert len(meshes) < 1000
    for i, mesh in enumerate(meshes):
        mesh_dir = paths.mesh_subdir(i)
        _save_mesh(mesh_dir, mesh)


def _load_meshes(mesh_dirs):
    meshes = []
    for mesh_dir in mesh_dirs:
        poly_geo_data = io_utils.load_json(mesh_dir / "geometry.json")
        polygons = [shapely_geo.shape(g) for g in poly_geo_data]

        arrays = io_utils.load_dict_of_arrays(mesh_dir / "arrays.npz")

        mesh = tutte_fields_core.Mesh(
            polygons=polygons,
            areas=arrays["areas"],
            anisotropies=arrays["anisotropies"],
        )
        meshes.append(mesh)

    return meshes


def _get_meshes(shape, paths):
    mesh_dirs = paths.mesh_subdirs
    if len(mesh_dirs) == 0:
        params = parameters.Params(shape=shape)
        meshes = tutte_fields_core.build_meshes(params)
        _save_meshes(paths, meshes)
    else:
        meshes = _load_meshes(mesh_dirs)
    return meshes


def _load_tutte_fields(path):
    data_dict = io_utils.load_dict_of_arrays(path)
    return tutte_fields_core.TutteFields(**data_dict)


def _generate_fields(shape, meshes):
    target_boundary = _get_general_target_boundary(shape)
    points_inside_shape = tutte_fields_core.get_points_inside_shape(
        target_boundary, nx=100, ny=100
    )

    area_field, anisotropy_field = tutte_fields_core.get_fields(
        meshes, points_inside_shape
    )

    tutte_fields_ = tutte_fields_core.TutteFields(
        points_inside_shape, area_field, anisotropy_field
    )
    return tutte_fields_


def get_fields(shape, paths):
    data_path = paths.fields_path(shape)
    if data_path.exists():
        tutte_fields_ = _load_tutte_fields(data_path)
    else:
        meshes = _get_meshes(shape, paths)

        tutte_fields_ = _generate_fields(shape, meshes)
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
