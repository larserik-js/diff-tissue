from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np

from ..core import init_systems, my_utils


def _get_polygons(vertices, indices, valid_mask):
    polygons_ = vertices[indices]
    polygons = [polygon[mask] for polygon, mask in zip(polygons_, valid_mask)]
    return polygons


class _Artists:
    def __init__(self, ax, init_vertices, outer_shape, all_knots, jax_arrays,
                 params):
        self._ax = ax
        self._init_vertices = init_vertices
        self._outer_shape = outer_shape
        self._ax_lims = self._get_ax_lims()
        self._all_knots = all_knots
        self._jax_arrays = jax_arrays
        self._params = params

    def _get_ax_lims(self):
        all_plotted_vertices = np.vstack(
            [self._init_vertices, self._outer_shape]
        )
        minvals = all_plotted_vertices.min(axis=0)
        maxvals = all_plotted_vertices.max(axis=0)
        center = init_systems.Coords.base_origin
        extrema = np.vstack([minvals, maxvals]) + center
        xlim = extrema[:,0] + np.array([-3.0, 3.0])
        ylim = extrema[:,1] + np.array([-2.0, 3.0])
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
            self._outer_shape[:, 0], self._outer_shape[:, 1],
            'ro-', markersize=3, label='Outer shape'
        )

    def _add_knots(self):
        self._ax.scatter(
            self._all_knots[:,0], self._all_knots[:,1],
            color='brown', s=20.0, alpha=1.0
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

    def _enumerate_centroids(self, vertices):
        centroids = my_utils.calc_centroids(
            vertices, self._jax_arrays['indices'],
            self._jax_arrays['valid_mask']
        )
        markers = np.arange(centroids.shape[0])
        for i, (x, y) in enumerate(centroids):
            self._ax.text(x, y, str(markers[i]), color='r')

    def _add_boundary_vertices(self, vertices):
        boundary_vertices = vertices[self._jax_arrays['boundary_mask']]
        self._ax.scatter(
            boundary_vertices[:,0], boundary_vertices[:,1], s=20.0, color='g',
            marker='s', zorder=3
        )

    def _add_artists(self, vertices, enumerate):
        self._add_baselines()
        self._add_outer_shape()
        self._add_vertices(vertices)
        self._add_boundary_vertices(vertices)
        if self._params.knots:
            self._add_knots()

        if enumerate:
            self._enumerate_centroids(vertices)

    def plot(self, vertices, enumerate):
        self._format()
        self._add_artists(vertices, enumerate)


class _Figure:
    def __init__(self, output_dir):
        self._output_dir = output_dir

    @staticmethod
    def _close(outer_shape):
        closed_outer_shape = np.vstack([outer_shape, outer_shape[0]])
        return closed_outer_shape

    def _save(self, step):
        fig_path = self._output_dir / f'step={step:03d}.png'
        self._fig.savefig(fig_path, dpi=100)


class MorphFigure(_Figure):
    def __init__(self, output_dir, jax_arrays, params):
        super().__init__(output_dir)
        self._fig, ax = plt.subplots(figsize=(10, 10))
        self._init_vertices = jax_arrays['init_vertices']
        self._closed_outer_shape = self._close(jax_arrays['outer_shape'])
        self._morph_artists = _Artists(
            ax, self._init_vertices, self._closed_outer_shape,
            jax_arrays['all_knots'], jax_arrays, params
        )

    def save_plot(self, vertices, step, enumerate=False):
        self._morph_artists.plot(vertices, enumerate)
        self._save(step)


class MorphGrowthFigure(_Figure):
    def __init__(self, output_dir, jax_arrays, params):
        super().__init__(output_dir)
        self._total_steps = params.n_growth_steps
        self._scale = params.growth_scale
        self._fig = plt.figure(figsize=(8, 10))
        self._gs = gridspec.GridSpec(
            nrows=2, ncols=1, figure=self._fig, height_ratios=[0.8, 1.0]
        )
        self._init_vertices = jax_arrays['init_vertices']
        self._closed_outer_shape = self._close(jax_arrays['outer_shape'])
        self._scaled_outer_shape = self._scale * self._closed_outer_shape
        self._all_knots = jax_arrays['all_knots']
        self._scaled_knots = self._scale * self._all_knots

        ax0 = self._fig.add_subplot(self._gs[0])
        self._morph_artists = _Artists(
            ax0, self._init_vertices, self._closed_outer_shape,
            self._all_knots, jax_arrays, params
        )
        ax1 = self._fig.add_subplot(self._gs[1:])
        self._growth_artists = _Artists(
            ax1, self._init_vertices, self._scaled_outer_shape,
            self._scaled_knots, jax_arrays, params
        )

    def _scale_vertices(self, vertices, step):
        t_frac = step / self._total_steps
        partial_scale = 1.0 + (self._scale - 1.0) * np.sin(0.5 * np.pi * t_frac)
        scaled_vertices = partial_scale * vertices
        return scaled_vertices

    def _plot(self, vertices, step, enumerate):
        self._morph_artists.plot(vertices, enumerate)

        scaled_vertices = self._scale_vertices(vertices, step)
        self._growth_artists.plot(scaled_vertices, enumerate)

    def save_plot(self, vertices, step, enumerate=False):
        self._plot(vertices, step, enumerate)
        self._save(step)
