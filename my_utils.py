import argparse
import os
from pathlib import Path
import pickle
import timeit

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

import init_systems


class _Output:
    _formats = {'bool': '',
                'int': 'd',
                'float': '.7f',
                'float64': '.7f',
                'str': ''}

    def __init__(self, output_type_dir, params):
        self._project_dir = self._get_project_dir()
        self._output_dir = self._project_dir / 'output'
        self._output_type_dir = self._output_dir / output_type_dir
        self._params = params
        self._param_path = self._make_param_path()

    def _get_project_dir(self):
        project_dir = os.path.abspath(os.path.dirname(__file__))
        return Path(project_dir)

    @staticmethod
    def _get_val_type(val):
        type_ = type(val)
        type_str = type_.__name__
        return type_str

    def _format_param_val_str(self, name, val):
        val_type = self._get_val_type(val)
        format_ = self._formats[val_type]
        param_name_val = name + '=' + format(val, format_)
        if val_type == 'float' or val_type == 'float64':
            param_name_val = param_name_val.rstrip('0').rstrip('.')
        return param_name_val

    def _concatenate_param_val_pairs(self):
        param_name_vals = []
        for name, val in self._params.all.items():
            param_name_val = self._format_param_val_str(name, val)
            param_name_vals.append(param_name_val)

        param_path_str = '_'.join(param_name_vals)
        return param_path_str

    def _make_param_path(self):
        param_path_str = self._concatenate_param_val_pairs()
        param_path = self._output_type_dir / param_path_str
        return param_path

    def get_output_type_dir(self):
        return self._output_type_dir


class OutputDir(_Output):
    def __init__(self, output_type_dir, params):
        super().__init__(output_type_dir, params)
        self._make()

    def _make(self):
        self._param_path.mkdir(exist_ok=True)

    def get_path(self):
        return self._param_path


class OutputFile(_Output):
    def __init__(self, output_type_dir, suffix, params):
        super().__init__(output_type_dir, params)
        self._path = self._param_path.with_name(self._param_path.name + suffix)

    def get_path(self):
        return self._path


class DataHandler:
    def __init__(self, file):
        self._file_path = file.get_path()

    def _load_pkl(self):
        with open(self._file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _save_pkl(self, data):
        with open(self._file_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        if self._file_path.suffix == '.pkl':
            data = self._load_pkl()
        else:
            raise NotImplementedError
        return data

    def save(self, data):
        if self._file_path.suffix == '.pkl':
            self._save_pkl(data)
        else:
            raise NotImplementedError


def get_output_params_file(params):
    output_params_file = (
        OutputDir('output_params', params).get_param_path_with_suffix('.txt')
    )
    return output_params_file


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
        'indices': polygons.get_polygon_inds(),
        'valid_mask': polygons.get_valid_mask(),
        'init_vertices': polygons.get_vertices(),
        'init_centroids': polygons.get_centroids(),
        'neighbors': polygons.get_neighbors(),
        'fixed_mask': polygons.get_fixed_mask(),
        'basal_mask': polygons.get_basal_mask(),
        'boundary_mask': polygons.get_boundary_mask(),
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
    polygons = factory.get_polygons()
    outer_shape = factory.get_outer_shape()
    arrays = _make_arrays(polygons, outer_shape)
    return arrays


def get_jax_arrays(params):
    arrays = get_arrays(params)
    jax_arrays = _make_jax_arrays(arrays)
    return jax_arrays


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


class Figure:
    def __init__(self, vertices):
        self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self._ax_lims = self._get_ax_lims(vertices)

    def _get_ax_lims(self, vertices):
        minvals = vertices.min(axis=0)
        maxvals = vertices.max(axis=0)
        center = init_systems.Coords.base_origin
        ranges = (maxvals - minvals)
        dims = np.array([0.8, 1.5]) * ranges
        xlim = center[0] + jnp.array([-1.0, 1.0]) * dims[0]
        ylim = center[1] + jnp.array([-1.0, dims[1]])
        ax_lims = {'x': xlim, 'y': ylim}
        return ax_lims

    def _format(self):
        self._ax.clear()
        self._ax.set_xlim(self._ax_lims['x'])
        self._ax.set_ylim(self._ax_lims['y'])
        self._ax.set_aspect('equal')

    def _add_baselines(self):
        baseline = np.block(
            [[self._ax_lims['x']], [init_systems.Coords.base_origin]]
        )

        self._ax.plot(baseline[0,:], baseline[1,:], 'k', lw=0.7)

    def _add_artists(self, vertices, jax_arrays):
        indices = jax_arrays['indices']
        for i in range(indices.shape[0]):
            vertex_inds = indices[i][jax_arrays['valid_mask'][i]]
            polygon = vertices[vertex_inds]
            self._ax.scatter(
                polygon[:, 0], polygon[:, 1], s=2.0, color='green', zorder=1
            )
            self._ax.plot(
                polygon[:, 0], polygon[:, 1], lw=0.7, color='black', zorder=2
            )

        self._add_baselines()

        boundary_vertices = vertices[jax_arrays['boundary_mask']]
        self._ax.scatter(
            boundary_vertices[:, 0], boundary_vertices[:, 1], s=20.0, color='g',
            marker='s', zorder=3
        )

        self._ax.plot(
            jax_arrays['outer_shape'][:, 0], jax_arrays['outer_shape'][:, 1],
            'ro-', markersize=3, label='Outer shape'
        )

    def _save(self, output_dir, step):
        fig_path = output_dir / f'step={step}.png'
        self._fig.savefig(fig_path, dpi=100)

    def plot(self, output_dir, vertices, jax_arrays, step):
        self._format()
        self._add_artists(vertices, jax_arrays)
        self._save(output_dir, step)
