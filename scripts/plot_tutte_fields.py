from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from diff_tissue import io_utils, tutte_fields


def _add_colorbar(ax, cmap_vals, cmap_name):
    normalize = colors.Normalize(
        vmin = np.nanmin(cmap_vals),
        vmax = np.nanmax(cmap_vals)
    )
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array(cmap_vals)
    ax.figure.colorbar(sm, ax=ax, shrink=0.8)


def _plot(tutte_fields):
    coords = tutte_fields.coords
    tutte_fields = [tutte_fields.areas, tutte_fields.anisotropies]

    titles = ['Tutte areas', 'Tutte anisotropies']
    cmaps = ['copper', 'viridis']

    fig, axs = plt.subplots(2, figsize=(5,6))
    for i, ax in enumerate(axs):
        tutte_field = tutte_fields[i]
        ax.scatter(
            coords[:,0], coords[:,1], c=tutte_field, cmap=cmaps[i], s=1.5
        )
        _add_colorbar(ax, tutte_field, cmaps[i])
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
    output_manager = io_utils.OutputManager(f'{tutte_fields.OUTPUT_TYPE_DIR}')
    output_file = output_manager.file_path(f'tutte_fields__{shape}.pdf')
    fig.savefig(output_file)


def _main():
    output = io_utils.OutputManager(tutte_fields.OUTPUT_TYPE_DIR)

    shape = 'petal'

    tutte_fields_ = tutte_fields.get_fields(shape, output)

    fig = _plot(tutte_fields_)

    _save_plot(fig, shape)


if __name__ == '__main__':
    _main()
