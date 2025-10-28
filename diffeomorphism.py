import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import init_systems, my_files, my_utils


def _cells_to_edges(polygons):
    edges = set()
    for polygon in polygons:
        # remove closing duplicate index if present
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon = polygon[:-1]
        for i, j in zip(polygon, polygon[1:] + polygon[:1]):
            if i != j:
                edges.add(tuple(sorted((i, j))))
    return list(edges)


def _tutte_embedding(n_vertices, edges, boundary_nodes, boundary_positions):
    free = jnp.setdiff1d(jnp.arange(n_vertices), boundary_nodes)

    L = scipy.sparse.lil_matrix((n_vertices, n_vertices))
    for i, j in edges:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
    L = L.tocsr()

    B = jnp.zeros((n_vertices, 2))

    B = B.at[boundary_nodes].set(boundary_positions)

    L_ff = L[free][:, free]
    L_fb = L[free][:, boundary_nodes]
    rhs = -L_fb @ B[boundary_nodes]

    U_free = scipy.sparse.linalg.spsolve(L_ff, rhs)
    UV = jnp.zeros((n_vertices, 2))
    UV = UV.at[boundary_nodes].set(B[boundary_nodes])
    UV = UV.at[free].set(U_free)
    return UV


def _rotate_rows(arr, k):
    k = int(k) % len(arr)
    return jnp.vstack([arr[k:], arr[:k]])


def _best_cyclic_shift(A, B):
    m = len(A)
    best_s, best_val = 0, float('inf')
    for s in range(m):
        Br = jnp.roll(B, -s, axis=0)
        val = jnp.sum((A - Br) ** 2)
        if val < best_val:
            best_val, best_s = val, s
    return best_s


def _map_to_given_shape(init_vertices, polygons, sorted_boundary_inds,
                        sorted_outer_shape, boundary_offset=0,
                        auto_align=False):
    # validate sizes
    m = len(sorted_boundary_inds)
    if len(sorted_outer_shape) != m:
        raise ValueError(
            f'boundary_target length {len(sorted_outer_shape)} != '
            + f'len(boundary_nodes) {m}'
        )

    edges = _cells_to_edges(polygons)

    # Step 1: embed initial to unit circle
    theta = 2 * jnp.pi * (jnp.arange(m) / m)
    circle_positions = jnp.column_stack([jnp.cos(theta), jnp.sin(theta)])
    UV_init = _tutte_embedding(
        len(init_vertices), edges, sorted_boundary_inds, circle_positions
    )

    # Step 2: rotate / align target boundary
    target_positions = _rotate_rows(sorted_outer_shape, boundary_offset)
    if auto_align:
        s = _best_cyclic_shift(circle_positions, target_positions)
        target_positions = _rotate_rows(sorted_outer_shape, boundary_offset + s)

    # Step 3: embed target shape
    mapped_positions = _tutte_embedding(
        len(init_vertices), edges, sorted_boundary_inds, target_positions
    )

    return UV_init, mapped_positions, edges


def _get_bottom_right_vertex(vertices):
    close_inds = jnp.where(
        jnp.isclose(
            vertices[:,1] - init_systems.Coords.base_origin[1], 0.0,
            atol=1.0
        )
    )
    close_vertices = vertices[close_inds]
    bottom_right_vertex = close_vertices[
        jnp.argmax(close_vertices[:,0])
    ]
    return bottom_right_vertex


def _get_bottom_right_idx(vertices):
    bottom_right_vertex = _get_bottom_right_vertex(vertices)
    bottom_right_idx = jnp.where(
        jnp.all(jnp.isclose(vertices - bottom_right_vertex, 0.0), axis=1)
    )[0][0]
    return bottom_right_idx


def _quick_plot(vertices):
    plt.scatter(vertices[:,0], vertices[:,1])
    for i, (x, y) in enumerate(vertices):
        plt.text(x + 0.02, y + 0.02, str(i), fontsize=9, color='red')
    plt.show()


def get_mapped_vertices(jax_arrays):
    init_vertices = jax_arrays['init_vertices']
    all_polygon_inds = jax_arrays['indices']
    boundary_mask = jax_arrays['boundary_mask']
    boundary_inds = jnp.where(boundary_mask)[0]
    boundary_vertices = init_vertices[boundary_inds]

    outer_shape = jax_arrays['outer_shape']

    sorted_boundary_inds = init_systems.sort_counterclockwise(
        boundary_inds, boundary_vertices
    )
    sorted_boundary_inds = jnp.array(sorted_boundary_inds)

    ccw_boundary_vertices = init_vertices[sorted_boundary_inds]

    bottom_right_idx = _get_bottom_right_idx(ccw_boundary_vertices)
    sorted_boundary_inds = jnp.roll(
        sorted_boundary_inds, -bottom_right_idx, axis=0
    )

    polygons = []
    for polygon_inds in all_polygon_inds:
        polygon = polygon_inds[polygon_inds != -1][:-2]
        polygons.append(polygon.tolist())

    sorted_inds = init_systems.sort_counterclockwise(
        jnp.arange(outer_shape.shape[0]), outer_shape
    )
    sorted_inds = jnp.array(sorted_inds)
    outer_shape = outer_shape[sorted_inds]
    bottom_right_idx = _get_bottom_right_idx(outer_shape)
    sorted_outer_shape = jnp.roll(outer_shape, -bottom_right_idx, axis=0)
    UV_init, mapped_vertices, edges = _map_to_given_shape(
        init_vertices, polygons, sorted_boundary_inds, sorted_outer_shape
    )

    return mapped_vertices


def _plot_mapping(output_file, jax_arrays, mapped_vertices):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    indices = jax_arrays['indices']
    init_vertices = jax_arrays['init_vertices']
    for ax, vertices, title in zip(axs, [init_vertices, mapped_vertices],
                                   ["Initial Mesh", "Mapped Mesh"]):
        for i in range(indices.shape[0]):
            vertex_inds = indices[i][jax_arrays['valid_mask'][i]]
            polygon = vertices[vertex_inds]
            ax.scatter(
                polygon[:, 0], polygon[:, 1], s=2.0, color='green', zorder=1
            )
            ax.plot(
                polygon[:, 0], polygon[:, 1], lw=0.7, color='black', zorder=2
            )
        ax.set_aspect('equal')
        ax.set_title(title)

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
    params = my_utils.Params()
    np.random.seed(params.numerical['seed'])
    jax_arrays = my_utils.get_jax_arrays(params)

    mapped_vertices = get_mapped_vertices(jax_arrays)

    output_file = my_files.OutputFile('diffeomorphism', '.pdf', params).path

    _plot_mapping(output_file, jax_arrays, mapped_vertices)


if __name__ == "__main__":
    _main()
