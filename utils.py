import argparse
import os
from pathlib import Path
import timeit

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt


def _get_project_dir():
    project_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    return project_dir


def get_output_dirs():
    project_dir = _get_project_dir()
    output_dirs = {'final_tissues': project_dir / 'final_tissues',
                   'best_growth': project_dir / 'best_growth',
                   'growth': project_dir / 'growth'}
    return output_dirs


def make_output_dirs():
    output_dirs = get_output_dirs()
    for output_dir in output_dirs.values():
        output_dir.mkdir(exist_ok=True)


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
        self.numerical_params = {
            'n_shape_steps': self._args.ssteps,
            'n_growth_steps': self._args.gsteps,
            'growth_learning_rate': self._args.glr,
            'areas_loss_weight': self._args.arlw,
            'angles_loss_weight': self._args.anlw,
            'aspect_ratio_loss_weight': self._args.aslw,
            'optimal_aspect_ratio': self._args.oar,
            'goal_area_weight': self._args.gaw,
            'max_area_scaling': self._args.marsc
        }

    def _parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--system',
            type=str,
            choices=['full', 'simple', 'voronoi'],
            default='simple',
            help='Initial polygon configuration.'
        )

        parser.add_argument(
            '--shape',
            type=str,
            choices=['ellipse', 'petal'],
            default='petal',
            help='Type of outer shape.'
        )

        parser.add_argument(
            '--ssteps',
            type=int,
            default=2000,
            help='Number of shape steps.'
        )

        parser.add_argument(
            '--gsteps',
            type=int,
            default=400,
            help='Number of growth steps.'
        )

        parser.add_argument(
            '--glr',
            type=float,
            default=0.0005,
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
            default=100.0,
            help='Angles loss weight.'
        )

        parser.add_argument(
            '--aslw',
            type=float,
            default=1.0,
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
            '--marsc',
            type=float,
            default=3.5,
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
