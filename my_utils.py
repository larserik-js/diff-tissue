import argparse
import os
from pathlib import Path
import timeit

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt


class _DirNames:
    _formats = {'bool': '',
                'int': 'd',
                'float': '.7f',
                'float64': '.7f',
                'str': ''}

    def __init__(self, params):
        self._project_dir = self._get_project_dir()
        self._output_dir = 'output'
        self._params = params
        self._param_dir = self._make_param_dir_name()

    def _get_project_dir(self):
        project_dir = os.path.abspath(os.path.dirname(__file__))
        return project_dir

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

    def _make_param_dir_name(self):
        param_name_vals = []
        for name, val in self._params.all.items():
            param_name_val = self._format_param_val_str(name, val)
            param_name_vals.append(param_name_val)

        param_dir = '_'.join(param_name_vals)
        return param_dir

    def _join_path(self, output_type_dir):
        full_path = (
            Path(self._project_dir) / Path(self._output_dir) /
            Path(output_type_dir) / Path(self._param_dir)
        )
        return full_path

    def get_output_dirs(self, output_type_dirs):
        output_dirs = {}
        for output_type_dir in output_type_dirs:
            output_type_dir = output_type_dir
            output_dir = self._join_path(output_type_dir)
            output_dirs[output_type_dir] = output_dir
        return output_dirs


class OutputDirs:
    def __init__(self, output_type_dirs, params):
        self._dir_names = _DirNames(params)
        self._output_dirs = self._dir_names.get_output_dirs(output_type_dirs)

    def make(self):
        for output_dir in self._output_dirs.values():
            output_dir.mkdir(parents=True, exist_ok=True)

    def get(self):
        return self._output_dirs


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
            'optimal_aspect_ratio': self._args.oar,
            'goal_area_weight': self._args.gaw,
            'goal_aspect_ratio_weight': self._args.gasw,
            'max_area_scaling': self._args.marsc
        }
        self.all = vars(self._args)

    def _parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--system',
            type=str,
            choices=['full', 'simple', 'voronoi', 'single'],
            default='simple',
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
            default=0.0002,
            help='Learning rate for growth.'
        )

        parser.add_argument(
            '--arlw',
            type=float,
            default=10.0,
            help='Areas loss weight.'
        )

        parser.add_argument(
            '--anlw',
            type=float,
            default=500.0,
            help='Angles loss weight.'
        )

        parser.add_argument(
            '--aslw',
            type=float,
            default=1e4,
            help='Aspect ratio loss weight.'
        )

        parser.add_argument(
            '--oar',
            type=float,
            default=1.0,
            help='Optimal aspect ratio.'
        )

        parser.add_argument(
            '--gaw',
            type=float,
            default=5e-5,
            help='Goal area weight.'
        )

        parser.add_argument(
            '--gasw',
            type=float,
            default=5e-5,
            help='Goal aspect ratio weight.'
        )

        parser.add_argument(
            '--marsc',
            type=float,
            default=5.0,
            help='Maximal area scaling.'
        )
        args = parser.parse_args()
        return args


def _get_device():
    return jax.devices('cpu')[0]


def _send_to_device(jax_arrays):
    return jax.device_put(jax_arrays, device=_get_device())


def get_jax_arrays(polygons, outer_shape):
    arrays = {
        'init_vertices': polygons.get_vertices(),
        'indices': polygons.get_polygon_inds(),
        'valid_mask': polygons.get_valid_mask(),
        'fixed_mask': polygons.get_fixed_mask(),
        'basal_mask': polygons.get_basal_mask(),
        'boundary_mask': polygons.get_boundary_mask(),
        'outer_shape': outer_shape
    }
    jax_arrays = {k: _send_to_device(jnp.array(v)) for k, v in arrays.items()}

    return jax_arrays


class Figure:
    def __init__(self, vertices):
        self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self._ax_lims = self._get_ax_lims(vertices)

    def _get_ax_lims(self, vertices):
        minvals = vertices.min(axis=0)
        maxvals = vertices.max(axis=0)
        center = (minvals + maxvals) / 2
        dims = maxvals - minvals
        xlim = center + jnp.array([-1.0, 1.0]) * dims[0]
        ylim = center + jnp.array([-1.0, 1.0]) * dims[1]
        return xlim, ylim

    def _format(self):
        self._ax.clear()
        self._ax.set_xlim(self._ax_lims[0])
        self._ax.set_ylim(self._ax_lims[1])
        self._ax.set_aspect('equal')

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

        base_y = 18.635
        self._ax.plot([-20, 10], [base_y, base_y], 'k', lw=0.7)
        self._ax.plot([70, 100], [base_y, base_y], 'k', lw=0.7)

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
