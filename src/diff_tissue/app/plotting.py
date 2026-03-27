from abc import ABC, abstractmethod
from functools import cached_property

from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib import figure as matplotlib_figure
import numpy as np

from ..core import init_systems, metrics, shapes


_GROWTH_SCALE = 5.0


def _get_polygons(vertices, indices, valid_mask):
    polygons_ = vertices[indices]
    polygons = [polygon[mask] for polygon, mask in zip(polygons_, valid_mask)]
    return polygons


class _Artists:
    def __init__(self, ax, polygons, target_boundary, all_knots, params):
        self._ax = ax
        self._polygons = polygons
        self._target_boundary = target_boundary
        self._all_knots = all_knots
        self._params = params
        self._ax_lims = self._get_ax_lims()

    def _get_ax_lims(self):
        all_plotted_vertices = np.vstack(
            [self._polygons.init_vertices, self._target_boundary]
        )
        minvals = all_plotted_vertices.min(axis=0)
        maxvals = all_plotted_vertices.max(axis=0)
        center = init_systems.Coords.base_origin
        extrema = np.vstack([minvals, maxvals]) + center
        xlim = extrema[:, 0] + np.array([-3.0, 3.0])
        ylim = extrema[:, 1] + np.array([-2.0, 3.0])
        ax_lims = {"x": xlim, "y": ylim}
        return ax_lims

    def _format(self):
        self._ax.clear()
        self._ax.set_xlim(self._ax_lims["x"])
        self._ax.set_ylim(self._ax_lims["y"])
        self._ax.set_aspect("equal")

    def _add_baselines(self):
        base_y = init_systems.Coords.base_origin[1]
        baseline = np.block([[self._ax_lims["x"]], [base_y, base_y]])
        self._ax.plot(baseline[0, :], baseline[1, :], "k", lw=0.7)

    def _add_target_boundary(self):
        self._ax.plot(
            self._target_boundary[:, 0],
            self._target_boundary[:, 1],
            "ro-",
            markersize=3,
            label="Target boundary",
        )

    def _add_knots(self):
        self._ax.scatter(
            self._all_knots[:, 0],
            self._all_knots[:, 1],
            color="brown",
            s=20.0,
            alpha=1.0,
        )

    def _add_vertices(self, vertices):
        polygons = _get_polygons(
            vertices,
            self._polygons.indices,
            self._polygons.valid_mask,
        )
        for polygon_ in polygons:
            polygon = polygon_[:-1]  # Removes redundant point
            self._ax.scatter(
                polygon[:, 0], polygon[:, 1], s=2.0, color="green", zorder=1
            )
            self._ax.plot(
                polygon[:, 0], polygon[:, 1], lw=0.7, color="black", zorder=2
            )

    def _enumerate_centroids(self, vertices):
        centroids = metrics.calc_centroids(
            vertices,
            self._polygons.indices,
            self._polygons.valid_mask,
        )
        markers = np.arange(centroids.shape[0])
        for i, (x, y) in enumerate(centroids):
            self._ax.text(x, y, str(markers[i]), color="r")

    def _add_boundary_vertices(self, vertices):
        boundary_vertices = vertices[self._polygons.boundary_inds]
        self._ax.scatter(
            boundary_vertices[:, 0],
            boundary_vertices[:, 1],
            s=20.0,
            color="g",
            marker="s",
            zorder=3,
        )

    def _add_artists(self, vertices, enumerate):
        self._add_baselines()
        self._add_target_boundary()
        self._add_vertices(vertices)
        self._add_boundary_vertices(vertices)
        if self._params.knots:
            self._add_knots()

        if enumerate:
            self._enumerate_centroids(vertices)

    def plot(self, vertices, enumerate):
        self._format()
        self._add_artists(vertices, enumerate)


class _Figure(ABC):
    def __init__(self, params):
        self._params = params
        self._polygons = init_systems.get_system(params)
        self._all_knots = init_systems.Knots().all_knots
        self._fig: matplotlib_figure.Figure
        self._init_figure()

    @abstractmethod
    def _init_figure(self):
        pass

    @staticmethod
    def _close(target_boundary):
        closed_target_boundary = np.vstack(
            [target_boundary, target_boundary[0]]
        )
        return closed_target_boundary

    @cached_property
    def _closed_target_boundary(self):
        vertex_numbers = init_systems.VertexNumbers(self._polygons)
        target_boundary = shapes.get_target_boundary(
            self._params, self._polygons.mesh_area, vertex_numbers
        ).vertices
        return self._close(target_boundary)

    def _save(self, fig_path):
        self._fig.savefig(fig_path, dpi=100)


class MorphFigure(_Figure):
    def __init__(self, params):
        super().__init__(params)
        ax = self._fig.add_subplot(111)
        self._morph_artists = _Artists(
            ax,
            self._polygons,
            self._closed_target_boundary,
            self._all_knots,
            params,
        )

    def _init_figure(self):
        self._fig = plt.figure(figsize=(10, 10))

    def save_plot(self, vertices, fig_path, enumerate=False):
        self._morph_artists.plot(vertices, enumerate)
        self._save(fig_path)


class MorphGrowthFigure(_Figure):
    def __init__(self, params):
        super().__init__(params)
        self._total_steps = params.n_morph_steps
        self._gs = gridspec.GridSpec(
            nrows=2, ncols=1, figure=self._fig, height_ratios=[0.8, 1.0]
        )
        self._scaled_target_boundary = (
            _GROWTH_SCALE * self._closed_target_boundary
        )
        self._scaled_knots = _GROWTH_SCALE * self._all_knots

        ax0 = self._fig.add_subplot(self._gs[0])
        self._morph_artists = _Artists(
            ax0,
            self._polygons,
            self._closed_target_boundary,
            self._all_knots,
            params,
        )
        ax1 = self._fig.add_subplot(self._gs[1:])
        self._growth_artists = _Artists(
            ax1,
            self._polygons,
            self._scaled_target_boundary,
            self._scaled_knots,
            params,
        )

    def _init_figure(self):
        self._fig = plt.figure(figsize=(8, 10))

    def _scale_vertices(self, vertices, step):
        t_frac = step / self._total_steps
        partial_scale = 1.0 + (_GROWTH_SCALE - 1.0) * np.sin(
            0.5 * np.pi * t_frac
        )
        scaled_vertices = partial_scale * vertices
        return scaled_vertices

    def _plot(self, vertices, step, enumerate):
        self._morph_artists.plot(vertices, enumerate)

        scaled_vertices = self._scale_vertices(vertices, step)
        self._growth_artists.plot(scaled_vertices, enumerate)

    def save_plot(self, vertices, step, fig_path, enumerate=False):
        self._plot(vertices, step, enumerate)
        self._save(fig_path)
