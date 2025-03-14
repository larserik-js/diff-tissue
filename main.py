import argparse
import jax
import jax.numpy as jnp
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi


_N_TIMESTEPS = 20000
_LEARNING_RATE = 6e-5

_AREAS_LOSS_WEIGHT = 10.0
_ANGLES_LOSS_WEIGHT = 100.0

_NUM_POLYGONS = 100
_MAX_VERTICES = 130


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        dest='mesh',
                        action='store_true',
                        help=('Use mesh as initial configuration.'))

    return parser.parse_args()


def _get_device():
    return jax.devices('cpu')[0]


def _get_project_dir():
    project_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    return project_dir


def _get_output_dir():
    project_dir = _get_project_dir()
    output_dir = project_dir / 'output'
    return output_dir


def _make_output_dir():
    output_dir = _get_output_dir()
    output_dir.mkdir(exist_ok=True, parents=True)


class _VoronoiPolygons:
    def __init__(self):
        self._all_polygon_vertex_inds, self._vertices = (
            self._make_init_polygons()
        )
        self._polygon_inds = self._finalize_polygon_inds()
        self._mask = (self._polygon_inds != -1)
        self._fixed_inds = np.array([], dtype=np.int64)

    def _is_finite(self, region):
        return (-1 not in region) and (len(region) > 0)

    def _inside_unit_square(self, vertices):
        inside_unit_square = (
            (vertices >= 0).all(axis=1) & (vertices <= 1).all(axis=1)
        )
        return inside_unit_square

    def _any_vertex_outside_unit_square(self, vertices):
        return np.any((vertices < 0) | (vertices > 1))

    def _extend_region(self, region):
        region.insert(0, region[-1])
        region.append(region[1])
        return region

    def _make_init_polygons(self):
        circumcenters = np.random.rand(_NUM_POLYGONS, 2)
        vor = Voronoi(circumcenters)
        inside_unit_square = self._inside_unit_square(vor.vertices)
        not_allowed_vertex_inds = np.where(~inside_unit_square)[0]
        allowed_vertices = vor.vertices[inside_unit_square]

        all_polygon_vertex_inds = []
        for region in vor.regions:
            if self._is_finite(region):
                vertices = vor.vertices[region]
                if self._any_vertex_outside_unit_square(vertices):
                    continue
            
                # For efficiency
                region = self._extend_region(region)
                vertex_inds = np.array(region)
                adjustment_inds = np.zeros_like(vertex_inds)
                for i in not_allowed_vertex_inds:
                    adjustment_inds -= (vertex_inds >= i).astype(int)
                vertex_inds += adjustment_inds
                all_polygon_vertex_inds.append(vertex_inds)

        return all_polygon_vertex_inds, allowed_vertices

    def _finalize_polygon_inds(self):
        all_polygon_inds = []
        for vertex_inds in self._all_polygon_vertex_inds:
            n_padding_values = _MAX_VERTICES - len(vertex_inds)
            padding_array = np.full((n_padding_values,), -1, dtype=np.long)
            polygon_inds = np.concatenate(
                [np.array(vertex_inds), padding_array]
            )
            all_polygon_inds.append(polygon_inds)
    
        all_polygon_inds = np.stack(all_polygon_inds)
        return all_polygon_inds

    def get_polygon_inds(self):
        return self._polygon_inds
    
    def get_mask(self):
        return self._mask
    
    def get_vertices(self):
        return self._vertices

    def get_fixed_inds(self):
        return self._fixed_inds


class _MeshPolygons:
    def __init__(self):
        self._input_cells = self._read_input_cells()
        self._all_polygon_vertex_inds, self._vertices, self._fixed_inds = (
            self._make_init_polygons()
        )
        self._mask = (self._all_polygon_vertex_inds != -1)
        self._fixed_inds = np.array([
            3, 0, 15, 16, 27, 37, 47, 66, 97, 103, 110, 145, 128, 123, 107, 78,
            52, 35, 18, 10, 11, 6, 7
        ])

    def _read_input_cells(self):
        input_path = Path('input_cells.json')
        with input_path.open() as data:
            input_cells = json.load(data)

        return input_cells

    def _make_init_polygons(self):
        all_vertices = np.zeros((0, 2))
        all_indices = []
        fixed_indices = []
        index = 0
        for polygon in self._input_cells:
            if polygon['is_boundary']:
                continue
            indices = []
            vertices = polygon['edges']
            for vertex in vertices:
                are_equal = np.isclose(
                    np.array(vertex) - all_vertices, 0.0, atol=0.5
                )
                possible_inds = np.where(np.all(are_equal, axis=1))[0]

                # Add new index
                if len(possible_inds) == 0:
                    all_vertices = np.vstack([all_vertices, vertex])
                    indices.append(index)
                    index += 1
                # Use existing index
                elif len(possible_inds) == 1:
                    indices.append(possible_inds[0])
                else:
                    raise ValueError('Multiple indices found')

            # For efficiency
            first_idx = indices[1]
            indices.append(first_idx)
            # Pad
            indices += [-1] * (_MAX_VERTICES - len(indices))
            indices.extend([-1] * (_MAX_VERTICES - len(indices)))
            all_indices.append(indices)
            if polygon['is_boundary']:
                fixed_indices.append(indices)

        all_indices = np.array(all_indices)
        fixed_indices = np.array(fixed_indices)

        return all_indices, all_vertices, fixed_indices

    def get_polygon_inds(self):
        return self._all_polygon_vertex_inds

    def get_mask(self):
        return self._mask

    def get_vertices(self):
        return self._vertices

    def get_fixed_inds(self):
        return self._fixed_inds


def _get_polygons(args):
    if args.mesh:
        polygons = _MeshPolygons()
    else:
        polygons = _VoronoiPolygons()
    return polygons


def _send_to_device(jax_arrays):
    return jax.device_put(jax_arrays, device=_get_device())


def _get_jax_arrays(polygons):
    vertices = jnp.array(polygons.get_vertices())
    indices = jnp.array(polygons.get_polygon_inds())
    mask = jnp.array(polygons.get_mask())
    fixed_inds = jnp.array(polygons.get_fixed_inds())

    jax_arrays = (vertices, indices, mask, fixed_inds)
    jax_arrays = _send_to_device(jax_arrays)

    return jax_arrays


def _update_target_areas(target_areas):
    areas_scales = 0.00005 * jnp.ones_like(target_areas)
    target_areas += areas_scales * target_areas
    return target_areas


def _calc_optimal_angles(mask):
    n_vertices = mask.sum(axis=1) - 2
    interior_angles = (n_vertices - 2) * jnp.pi / n_vertices
    optimal_angles = jnp.pi - interior_angles
    optimal_angles = optimal_angles[:, None]
    return optimal_angles


def _calc_all_areas(all_cells, mask):
    xs = all_cells[:, 1:-1, 0]
    y_plus_ones = all_cells[:, 2:, 1]
    y_minus_ones = all_cells[:, :-2, 1]

    valid = mask[:, 1:-1] & mask[:, 2:] & mask[:, :-2]

    first_term = xs * y_plus_ones
    first_term = jnp.sum(first_term * valid, axis=1)
    second_term = xs * y_minus_ones
    second_term = jnp.sum(second_term * valid, axis=1)

    # Abs. because vertex orientation can be
    # both clockwise and counter-clockwise
    areas = 0.5 * jnp.abs(first_term - second_term)

    return areas


def _calc_all_angles_loss(all_cells, mask, optimal_angles):
    epsilon = 1e-7
    edges = all_cells[:, 1:] - all_cells[:, :-1]
    dot_products = jnp.sum(edges[:, :-1] * edges[:, 1:], axis=2)
    norms = jnp.linalg.norm(edges + epsilon, axis=2)
    cosines = dot_products / (epsilon + norms[:, :-1] * norms[:, 1:])
    clip_value = 1.0 - epsilon
    cosines = jnp.clip(cosines, -clip_value, clip_value)
    angles = jnp.arccos(cosines)

    valid = mask[:, 1:] & mask[:, :-1]
    valid = valid[:, 1:] & valid[:, :-1]
    angles_loss = jnp.sum((angles - optimal_angles)**2 * valid)

    return angles_loss


def _calc_loss(vertices, indices, mask, target_areas, optimal_angles):
    all_cells = vertices[indices]
    areas = _calc_all_areas(all_cells, mask)
    areas_loss = _AREAS_LOSS_WEIGHT * jnp.sum((target_areas - areas)**2)
    angles_loss = _ANGLES_LOSS_WEIGHT * _calc_all_angles_loss(
        all_cells, mask, optimal_angles
    )

    jax.debug.print('Areas loss: {}', areas_loss)
    jax.debug.print('Angles loss: {}', angles_loss)
    jax.debug.print('')

    loss = areas_loss + angles_loss

    return loss


def _get_ax_lims(vertices):
    minvals = vertices.min(axis=0)
    maxvals = vertices.max(axis=0)
    center = (minvals + maxvals) / 2
    dims = maxvals - minvals
    xlim = center + jnp.array([-1.0, 1.0]) * dims[0]
    ylim = center + jnp.array([-1.0, 1.0]) * dims[1]
    return xlim, ylim


def _format(ax, xlim, ylim):
    ax.clear()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')


def _plot(ax, vertices, indices, mask):
    for i in range(indices.shape[0]):
        vertex_inds = indices[i][mask[i]]
        polygon = vertices[vertex_inds]
        ax.scatter(polygon[:, 0], polygon[:, 1], s=2.0, color='green', zorder=1)
        ax.plot(polygon[:, 0], polygon[:, 1], lw=0.7, color='black', zorder=2)

    base_y = 18.635
    ax.plot([-20, 10], [base_y, base_y], 'k', lw=0.7)
    ax.plot([70, 100], [base_y, base_y], 'k', lw=0.7)
    

def _save_figure(fig, step):
    output_dir = _get_output_dir()
    fig_path = output_dir / f'step_{step}.png'
    
    fig.savefig(fig_path, dpi=100)


def _iterate(vertices, indices, mask, fixed_inds):
    fig, ax = plt.subplots(figsize=(10, 10))
    xlim, ylim = _get_ax_lims(vertices)

    all_cells = vertices[indices]
    target_areas = _calc_all_areas(all_cells, mask)
    optimal_angles = _calc_optimal_angles(mask)

    _calc_loss_and_grads = jax.value_and_grad(_calc_loss)
    _calc_loss_and_grads = jax.jit(_calc_loss_and_grads)

    for t in range(_N_TIMESTEPS):
        target_areas = _update_target_areas(target_areas)
        loss, grads = _calc_loss_and_grads(
            vertices, indices, mask, target_areas, optimal_angles
        )
        grads = grads.at[fixed_inds].set(0.0)
        vertices -= _LEARNING_RATE * grads

        if t % int(_N_TIMESTEPS / 100) == 0:
            print(f'Step {t}, Loss: {loss}')
        
            _format(ax, xlim, ylim)
            _plot(ax, vertices, indices, mask)
            _save_figure(fig, t)


def _main():
    np.random.seed(0)
    jax.config.update('jax_enable_x64', True)

    _make_output_dir()

    args = parse_args()

    polygons = _get_polygons(args)

    vertices, indices, mask, fixed_inds = _get_jax_arrays(polygons)

    _iterate(vertices, indices, mask, fixed_inds)


if __name__ == "__main__":
    _main()
