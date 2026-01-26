import pathlib
import pickle

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


def _load_fields(input_file):
    with open(input_file, 'rb') as f:
        fields = pickle.load(f)
    return fields


def _add_colorbar(ax, cmap_vals, cmap_name):
    normalize = colors.Normalize(
        vmin = np.nanmin(cmap_vals),
        vmax = np.nanmax(cmap_vals)
    )
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array(cmap_vals)
    ax.figure.colorbar(sm, ax=ax, shrink=0.8)


def _plot(*, grid_coords, area_field_grid, elongation_field_grid):
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


def _save_plot(fig):
    output_file = pathlib.Path('outputs/fields.pdf')
    output_file.parent.mkdir(exist_ok=True)
    fig.savefig(output_file)


def _main():
    fields_file = pathlib.Path('outputs/fields.pkl')
    fields = _load_fields(fields_file)

    fig = _plot(**fields)

    _save_plot(fig)


if __name__ == '__main__':
    _main()
