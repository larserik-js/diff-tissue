import argparse
import timeit

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

import init_systems


def timer(func):
    def timed(*args, **kwargs):
        t_init = timeit.default_timer()
        res = func(*args, **kwargs)
        t_end = timeit.default_timer()

        t_tot = t_end - t_init

        print(f'Total time: {t_tot:.4f} s')
        return res

    return timed


class Params:
    def __init__(self):
        self._args = self._parse_args()
        self.system = self._args.system
        self.shape = self._args.shape
        self.numerical = {
            'n_shape_steps': self._args.ssteps,
            'n_growth_steps': self._args.gsteps,
            'growth_learning_rate': self._args.glr,
            'areas_loss_weight': self._args.arlw,
            'angles_loss_weight': self._args.anlw,
            'aspect_ratio_loss_weight': self._args.aslw,
            'max_area_scaling': self._args.marsc,
            'seed': self._args.seed
        }
        self.all = vars(self._args)

    def _parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--system',
            type=str,
            choices=['full', 'voronoi', 'single'],
            default='voronoi',
            help='Initial polygon configuration.'
        )

        parser.add_argument(
            '--shape',
            type=str,
            choices=['ellipse', 'trapezoid', 'petal'],
            default='petal',
            help='Type of outer shape.'
        )

        parser.add_argument(
            '--ssteps',
            type=int,
            default=500,
            help='Number of shape steps.'
        )

        parser.add_argument(
            '--gsteps',
            type=int,
            default=500,
            help='Number of growth steps.'
        )

        parser.add_argument(
            '--glr',
            type=float,
            default=0.00005,
            help='Learning rate for growth.'
        )

        parser.add_argument(
            '--arlw',
            type=float,
            default=100.0,
            help='Areas loss weight.'
        )

        parser.add_argument(
            '--anlw',
            type=float,
            default=100.0,
            help='Angles loss weight.'
        )

        parser.add_argument(
            '--aslw',
            type=float,
            default=1000.0,
            help='Aspect ratio loss weight.'
        )

        parser.add_argument(
            '--marsc',
            type=float,
            default=3.0,
            help='Maximal area scaling.'
        )

        parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='Random NumPy seed for reproducibility.'
        )
        args = parser.parse_args()
        return args


def _make_arrays(polygons, outer_shape):
    arrays = {
        'indices': polygons.polygon_inds,
        'valid_mask': polygons.valid_mask,
        'init_vertices': polygons.vertices,
        'init_centroids': polygons.centroids,
        'poly_neighbors': polygons.poly_neighbors,
        'vertex_neighbors': polygons.vertex_neighbors,
        'free_mask': polygons.free_mask,
        'proximal_mask': polygons.proximal_mask,
        'boundary_mask': polygons.boundary_mask,
        'outer_shape': outer_shape
    }
    return arrays


def _get_device():
    return jax.devices('cpu')[0]


def _send_to_device(jax_array):
    return jax.device_put(jax_array, device=_get_device())


def to_jax(np_array):
    return _send_to_device(jnp.array(np_array))


def _make_jax_arrays(arrays):
    jax_arrays = {name: to_jax(array) for name, array in arrays.items()}
    return jax_arrays


def get_arrays(params):
    factory = init_systems.get_factory(params.shape, params.system)
    polygons = factory.polygons
    outer_shape = factory.outer_shape
    arrays = _make_arrays(polygons, outer_shape)
    return arrays


def get_jax_arrays(params):
    arrays = get_arrays(params)
    jax_arrays = _make_jax_arrays(arrays)
    return jax_arrays


def calc_optimal_angles(valid_mask):
    n_vertices = valid_mask.sum(axis=1) - 2
    interior_angles = (n_vertices - 2) * jnp.pi / n_vertices
    optimal_angles = jnp.pi - interior_angles
    optimal_angles = optimal_angles[:, None]
    return optimal_angles


def get_all_cells(vertices, indices):
    all_cells = vertices[indices]
    return all_cells


def _get_polygons(vertices, indices, valid_mask):
    polygons_ = vertices[indices]
    polygons = [polygon[mask] for polygon, mask in zip(polygons_, valid_mask)]
    return polygons


def calc_all_areas(all_cells, valid_mask):
    xs = all_cells[:, 1:-1, 0]
    y_plus_ones = all_cells[:, 2:, 1]
    y_minus_ones = all_cells[:, :-2, 1]

    valid = valid_mask[:, 1:-1] & valid_mask[:, 2:] & valid_mask[:, :-2]

    first_term = xs * y_plus_ones
    first_term = jnp.sum(first_term * valid, axis=1)
    second_term = xs * y_minus_ones
    second_term = jnp.sum(second_term * valid, axis=1)

    # Assumes vertices are ordered counter-clockwise
    areas = 0.5 * (first_term - second_term)

    return areas


def _masked_min(values, mask):
    masked_values = jnp.where(mask, values, jnp.inf)
    return jnp.min(masked_values, axis=1)


def _masked_max(values, mask):
    masked_values = jnp.where(mask, values, -jnp.inf)
    return jnp.max(masked_values, axis=1)


def calc_aspect_ratios(all_cells, valid_mask):
    min_xys = _masked_min(all_cells, valid_mask[:, :, None])
    max_xys = _masked_max(all_cells, valid_mask[:, :, None])

    widths = max_xys[:,0] - min_xys[:,0]
    heights = max_xys[:,1] - min_xys[:,1]

    aspect_ratios = widths / (heights + widths)

    return aspect_ratios


class _Artists:
    def __init__(self, ax, init_vertices, outer_shape, jax_arrays):
        self._ax = ax
        self._jax_arrays = jax_arrays
        self._init_vertices = init_vertices
        self._closed_outer_shape = self._close(outer_shape)
        self._boundary_mask = jax_arrays['boundary_mask']
        self._ax_lims = self._get_ax_lims()

    @staticmethod
    def _close(outer_shape):
        closed_outer_shape = np.vstack([outer_shape, outer_shape[0]])
        return closed_outer_shape

    def _get_ax_lims(self):
        minvals = self._init_vertices.min(axis=0)
        maxvals = self._init_vertices.max(axis=0)
        center = init_systems.Coords.base_origin
        ranges = (maxvals - minvals)
        dims = np.array([0.8, 1.5]) * ranges
        xlim = center[0] + np.array([-1.0, 1.0]) * dims[0]
        ylim = center[1] + np.array([-1.0, dims[1]])
        ax_lims = {'x': xlim, 'y': ylim}
        return ax_lims

    def _format(self):
        self._ax.clear()
        self._ax.set_xlim(self._ax_lims['x'])
        self._ax.set_ylim(self._ax_lims['y'])
        self._ax.set_aspect('equal')

    def _add_baselines(self):
        base_y = init_systems.Coords.base_origin[1]
        baseline = np.block(
            [[self._ax_lims['x']], [base_y, base_y]]
        )
        self._ax.plot(baseline[0,:], baseline[1,:], 'k', lw=0.7)

    def _add_outer_shape(self):
        self._ax.plot(
            self._closed_outer_shape[:, 0], self._closed_outer_shape[:, 1],
            'ro-', markersize=3, label='Outer shape'
        )

    def _add_vertices(self, vertices):
        polygons = _get_polygons(
            vertices, self._jax_arrays['indices'],
            self._jax_arrays['valid_mask']
        )
        for polygon_ in polygons:
            polygon = polygon_[:-1] # Removes redundant point
            self._ax.scatter(
                polygon[:,0], polygon[:,1], s=2.0, color='green', zorder=1
            )
            self._ax.plot(
                polygon[:,0], polygon[:,1], lw=0.7, color='black', zorder=2
            )

    def _add_boundary_vertices(self, vertices):
        boundary_vertices = vertices[self._jax_arrays['boundary_mask']]
        self._ax.scatter(
            boundary_vertices[:,0], boundary_vertices[:,1], s=20.0, color='g',
            marker='s', zorder=3
        )

    def _add_artists(self, vertices):
        self._add_baselines()
        self._add_outer_shape()
        self._add_vertices(vertices)
        self._add_boundary_vertices(vertices)

    def plot(self, vertices):
        self._format()
        self._add_artists(vertices)


class _Figure:
    def __init__(self, output_dir):
        self._output_dir = output_dir

    def _save(self, step):
        fig_path = self._output_dir / f'step={step}.png'
        self._fig.savefig(fig_path, dpi=100)


class MorphFigure(_Figure):
    def __init__(self, output_dir, jax_arrays):
        super().__init__(output_dir)
        self._fig, ax = plt.subplots(figsize=(10, 10))
        self._artists = _Artists(
            ax, jax_arrays['init_vertices'], jax_arrays['outer_shape'],
            jax_arrays
        )

    def save_plot(self, vertices, step):
        self._artists.plot(vertices)
        self._save(step)


class MorphGrowthFigure(_Figure):
    def __init__(self, output_dir, jax_arrays, total_steps, scale):
        super().__init__(output_dir)
        self._total_steps = total_steps
        self._scale = scale
        self._fig, axs = plt.subplots(2, figsize=(10, 10))
        self._morph_artists = _Artists(
            axs[0], jax_arrays['init_vertices'], jax_arrays['outer_shape'],
            jax_arrays
        )
        self._growth_artists = _Artists(
            axs[1], scale * jax_arrays['init_vertices'],
            scale * jax_arrays['outer_shape'], jax_arrays
        )

    def _scale_vertices(self, vertices, step):
        t_frac = step / self._total_steps
        partial_scale = 1.0 + t_frac * (self._scale - 1.0)
        scaled_vertices = partial_scale * vertices
        return scaled_vertices

    def _plot(self, vertices, step):
        self._morph_artists.plot(vertices)

        scaled_vertices = self._scale_vertices(vertices, step)
        self._growth_artists.plot(scaled_vertices)

    def save_plot(self, vertices, step):
        self._plot(vertices, step)
        self._save(step)
