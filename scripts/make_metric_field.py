from dataclasses import dataclass
import pathlib
import pickle

import jax
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.strtree import STRtree

from diff_tissue import my_utils


@dataclass
class _Mesh:
    polygons: list
    areas: np.ndarray
    elongations: np.ndarray


def _get_outer_shape():
    np.random.seed(0)
    params = my_utils.Params()
    jax_arrays = my_utils.get_jax_arrays(params)
    outer_shape = jax_arrays['outer_shape']
    return outer_shape


def _make_samples(nx, ny, outer_shape):
    xmin, ymin = outer_shape.min(axis=0)
    xmax, ymax = outer_shape.max(axis=0)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    all_points = np.column_stack([X.ravel(), Y.ravel()])

    return all_points


def _get_mapped_metrics(jax_arrays):
    all_mapped_cells = my_utils.get_all_cells(
        jax_arrays['mapped_vertices'], jax_arrays['indices']
    )
    mapped_areas = my_utils.calc_all_areas(
        all_mapped_cells, jax_arrays['valid_mask']
    )
    mapped_elongations = my_utils.calc_elongations(
        all_mapped_cells, jax_arrays['valid_mask']
    )
    return mapped_areas, mapped_elongations


def _build_meshes(n_meshes, output_file):
    if output_file.exists():
        with open(output_file, 'rb') as f:
            meshes = pickle.load(f)
        return meshes
    else:
        params = my_utils.Params()
        meshes = []

        for i in range(n_meshes):
            print(i)
            params.numerical['seed'] = i
            np.random.seed(params.numerical['seed'])
            jax_arrays = my_utils.get_jax_arrays(params)
            shapely_polygons = my_utils.get_shapely_polygons(
                jax_arrays['mapped_vertices'], jax_arrays['indices']
            )
            mapped_areas, mapped_elongations = _get_mapped_metrics(jax_arrays)

            mesh = _Mesh(
                shapely_polygons, np.array(mapped_areas),
                np.array(mapped_elongations)
            )
            meshes.append(mesh)
    
        with open(output_file, 'wb') as f:
            pickle.dump(meshes, f)
    
    return meshes


def _sample_mesh(mesh: _Mesh, pts_np: np.ndarray):
    """
    Assign mesh scalar values to sample points.
    Points must already be NumPy and lie inside the domain.
    """

    tree = STRtree(mesh.polygons)
    geom_pts = shapely.points(pts_np)

    sampled_areas = np.full(len(pts_np), np.nan)
    sampled_elongations = np.full(len(pts_np), np.nan)

    # IMPORTANT:
    # predicate must be 'intersects'
    # index order is (point_index, polygon_index)
    matches = tree.query(geom_pts, predicate='intersects')

    pt_indices, poly_indices = matches
    sampled_areas[pt_indices] = mesh.areas[poly_indices]
    sampled_elongations[pt_indices] = mesh.elongations[poly_indices]

    return sampled_areas, sampled_elongations


def _aggregate_meshes(meshes, pts_np):
    """
    Sample all meshes and average their scalar fields.
    """
    all_sampled_areas = []
    all_sampled_elongations = []

    for mesh in meshes:
        sampled_areas, sampled_elongations = _sample_mesh(mesh, pts_np)
        all_sampled_areas.append(sampled_areas)
        all_sampled_elongations.append(sampled_elongations)

    stacked_areas = np.vstack(all_sampled_areas)
    stacked_elongations = np.vstack(all_sampled_elongations)

    mean_areas = np.nanmean(stacked_areas, axis=0)
    mean_elongations = np.nanmean(stacked_elongations, axis=0)

    return mean_areas, mean_elongations


def _save_fields(grid_coords, area_field_grid, elongation_field_grid,
                 output_file):
    output = {
        'grid_coords': grid_coords,
        'area_field_grid': area_field_grid,
        'elongation_field_grid': elongation_field_grid
    }
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)


def _add_colorbar(ax, cmap_vals, cmap_name):
    normalize = colors.Normalize(
        vmin = np.nanmin(cmap_vals),
        vmax = np.nanmax(cmap_vals)
    )
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array(cmap_vals)
    ax.figure.colorbar(sm, ax=ax, shrink=0.8)


def _plot(grid_coords, area_field_grid, elongation_field_grid):
    xmin, ymin = grid_coords.min(axis=0)
    xmax, ymax = grid_coords.max(axis=0)

    field_grids = [area_field_grid, elongation_field_grid]

    titles = ['Goal areas', 'Goal elongations']
    cmaps = ['copper', 'viridis']

    fig, axs = plt.subplots(2, figsize=(5,6))
    for i, ax in enumerate(axs):
        field_grid = field_grids[i]
        ax.imshow(
            field_grid, cmap=cmaps[i], extent=(xmin, xmax, ymin, ymax),
            origin='lower', aspect='auto',
        )
        _add_colorbar(ax, field_grid, cmaps[i])
        ax.set_title(titles[i])
        ax.set_aspect('equal')
        ax.set_xlim(-11.0, 11.0)
        ax.set_ylim(-0.5, 16.1)
        if i == 0:
            ax.set_xticklabels([])
        if i == 1:
            ax.set_xlabel('$x$')

    fig.tight_layout()
    return fig


def _to_grid(field, grid_coords, mask, nx, ny):
    field_full = np.full(len(grid_coords), np.nan)
    field_full[mask] = field
    field_grid = field_full.reshape((ny, nx))
    return field_grid


def _save_plot(fig):
    output_file = pathlib.Path('outputs/fields.pdf')
    output_file.parent.mkdir(exist_ok=True)
    fig.savefig(output_file)


def _main():
    jax.config.update('jax_enable_x64', True)

    outer_shape = _get_outer_shape()

    nx, ny = 100, 100
    grid_coords = _make_samples(nx, ny, outer_shape)

    domain_polygon = shapely.Polygon(outer_shape)
    pts_geom = shapely.points(grid_coords)
    mask = domain_polygon.covers(pts_geom)
    filtered_pts = grid_coords[mask]

    n_meshes = 100

    output_dir = pathlib.Path('outputs')
    output_dir.mkdir(exist_ok=True)

    meshes_file = output_dir / 'meshes.pkl'
    meshes = _build_meshes(n_meshes, meshes_file)

    area_field, elongation_field = _aggregate_meshes(meshes, filtered_pts)

    area_field_grid = _to_grid(area_field, grid_coords, mask, nx, ny)
    elongation_field_grid = _to_grid(
        elongation_field, grid_coords, mask, nx, ny
    )

    fields_file = output_dir / 'fields.pkl'
    _save_fields(
        grid_coords, area_field_grid, elongation_field_grid, fields_file
    )

    fig = _plot(grid_coords, area_field_grid, elongation_field_grid)

    _save_plot(fig)


if __name__ == '__main__':
    _main()
