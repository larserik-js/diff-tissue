import pathlib
import pickle

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from diff_tissue import io_utils


def _load_mapped_fields(input_file):
    with open(input_file, 'rb') as f:
        mapped_fields = pickle.load(f)
    return mapped_fields


def _add_colorbar(ax, cmap_vals, cmap_name):
    normalize = colors.Normalize(
        vmin = np.nanmin(cmap_vals),
        vmax = np.nanmax(cmap_vals)
    )
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array(cmap_vals)
    ax.figure.colorbar(sm, ax=ax, shrink=0.8)


def _plot(*, coords, mapped_area_field, mapped_elongation_field):
    mapped_fields = [mapped_area_field, mapped_elongation_field]

    titles = ['Mapped areas', 'Mapped elongations']
    cmaps = ['copper', 'viridis']

    fig, axs = plt.subplots(2, figsize=(5,6))
    for i, ax in enumerate(axs):
        mapped_field = mapped_fields[i]
        ax.scatter(
            coords[:,0], coords[:,1], c=mapped_field,
            cmap=cmaps[i], s=1.5
        )
        _add_colorbar(ax, mapped_field, cmaps[i])
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


def _save_plot(fig, shape):
    output_file = io_utils.get_output_path(f'mapped_fields_{shape}.pdf')
    fig.savefig(output_file)


def _main():
    shape = 'petal'
    mapped_fields_file = io_utils.get_output_path(f'mapped_fields_{shape}.pkl')
    mapped_fields = _load_mapped_fields(mapped_fields_file)

    fig = _plot(**mapped_fields)

    _save_plot(fig, shape)


if __name__ == '__main__':
    _main()
