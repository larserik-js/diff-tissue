import argparse
import timeit

import jax
import jax.numpy as jnp

import init_systems, my_files, shapes


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
            'areas_loss_weight': self._args.arlw,
            'angles_loss_weight': self._args.anlw,
            'aspect_ratio_loss_weight': self._args.aslw,
            'max_area_scaling': self._args.marsc,
            'growth_scale': self._args.gsc,
            'seed': self._args.seed
        }
        self.all = vars(self._args)

    def _parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--system',
            type=str,
            choices=['voronoi', 'full', 'single'],
            default='voronoi',
            help='Initial polygon configuration.'
        )

        parser.add_argument(
            '--shape',
            type=str,
            choices=['petal', 'trapezoid', 'triangle', 'nconv'],
            default='petal',
            help='Type of outer shape.'
        )

        parser.add_argument(
            '--ssteps',
            type=int,
            default=100,
            help='Number of shape steps.'
        )

        parser.add_argument(
            '--gsteps',
            type=int,
            default=100,
            help='Number of growth steps.'
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
            default=70.0,
            help='Angles loss weight.'
        )

        parser.add_argument(
            '--aslw',
            type=float,
            default=200.0,
            help='Aspect ratio loss weight.'
        )

        parser.add_argument(
            '--marsc',
            type=float,
            default=9.0,
            help='Maximal area scaling.'
        )

        parser.add_argument(
            '--gsc',
            type=float,
            default=5.0,
            help='Growth scale.'
        )

        parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='Random NumPy seed for reproducibility.'
        )
        args = parser.parse_args()
        return args


def _make_array_dict(polygons, outer_shape):
    arrays = {
        'indices': polygons.polygon_inds,
        'valid_mask': polygons.valid_mask,
        'init_vertices': polygons.vertices,
        'poly_neighbors': polygons.poly_neighbors,
        'vertex_neighbors': polygons.vertex_neighbors,
        'vertex_polygons': polygons.vertex_polygons,
        'free_mask': polygons.free_mask,
        'proximal_mask': polygons.proximal_mask,
        'boundary_mask': polygons.boundary_mask,
        'outer_shape': outer_shape
    }
    return arrays


def _get_arrays(params):
    polygons = init_systems.get_system(params.system)
    outer_shape = shapes.get_outer_shape(params.shape, polygons)
    arrays = _make_array_dict(polygons, outer_shape)
    return arrays


def get_arrays(params):
    file = my_files.ArraysFile('arrays', '.pkl', params)
    data_handler = my_files.DataHandler(file)

    try:
        arrays = data_handler.load()
    except:
        arrays = _get_arrays(params)
        data_handler.save(arrays)
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


def calc_aspect_ratios(all_cells, valid_mask):
    xs = all_cells[:, 1:-1, 0]
    ys = all_cells[:, 1:-1, 1]
    valid = valid_mask[:, 1:-1]

    xs_masked = jnp.where(valid, xs, jnp.nan)
    ys_masked = jnp.where(valid, ys, jnp.nan)

    x_vars = jnp.nanvar(xs_masked, axis=1)
    y_vars = jnp.nanvar(ys_masked, axis=1)

    eps = 1e-8
    aspect_ratios = (y_vars - x_vars) / (y_vars + x_vars + eps)

    return aspect_ratios


def calc_centroids(vertices, indices, valid_mask):
    polygons = vertices[indices]
    mask = valid_mask[..., None].repeat(2, axis=2)
    polygons = jnp.where(mask, polygons, jnp.nan)
    centroids = jnp.nanmean(polygons, axis=1)
    return centroids
