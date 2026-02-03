import matplotlib.pyplot as plt
import numpy as np

from diff_tissue import my_files, my_utils, parameters


def _add_artists(ax, jax_arrays, vertices):
    for i in range(jax_arrays['indices'].shape[0]):
        vertex_inds = jax_arrays['indices'][i][jax_arrays['valid_mask'][i]]
        polygon = vertices[vertex_inds]
        ax.scatter(
            polygon[:, 0], polygon[:, 1], s=2.0, color='green', zorder=1
        )
        ax.plot(
            polygon[:, 0], polygon[:, 1], lw=0.7, color='black', zorder=2
        )


def _plot_mapping(output_file, jax_arrays):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    # Initial mesh
    ax = axs[0]
    init_vertices = jax_arrays['init_vertices']
    _add_artists(ax, jax_arrays, init_vertices)
    ax.set_aspect('equal')
    ax.set_title('Initial mesh')

    # Mapped mesh
    ax = axs[1]
    mapped_vertices = jax_arrays['mapped_vertices']
    _add_artists(ax, jax_arrays, mapped_vertices)
    ax.set_aspect('equal')
    ax.set_title('Mapped mesh')

    # Vector field from initial to mapped
    ax = axs[2]
    ax.quiver(init_vertices[:, 0], init_vertices[:, 1],
              mapped_vertices[:, 0] - init_vertices[:, 0],
              mapped_vertices[:, 1] - init_vertices[:, 1],
              angles='xy', scale_units='xy', scale=1.0, color='r')

    offset = 3.5
    minvals = mapped_vertices.min(axis=0)
    maxvals = mapped_vertices.max(axis=0)
    xlim = np.array([minvals[0] - offset, maxvals[0] + offset])
    ylim = np.array([minvals[1] - offset, maxvals[1] + offset])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title("Vector Field: Initial → Mapped")

    fig.tight_layout()
    fig.savefig(output_file)


def _main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    output_file = my_files.OutputFile('diffeomorphism', '.pdf', params).path

    _plot_mapping(output_file, jax_arrays)


if __name__ == "__main__":
    _main()
