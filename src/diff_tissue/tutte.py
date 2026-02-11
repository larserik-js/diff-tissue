import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from . import init_systems


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
    free = np.setdiff1d(np.arange(n_vertices), boundary_nodes)

    L = scipy.sparse.lil_matrix((n_vertices, n_vertices))
    for i, j in edges:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
    L = L.tocsr()

    B = np.zeros((n_vertices, 2))
    B[boundary_nodes] = boundary_positions

    L_ff = L[free][:, free]
    L_fb = L[free][:, boundary_nodes]
    rhs = -L_fb @ B[boundary_nodes]

    U_free = scipy.sparse.linalg.spsolve(L_ff, rhs)
    UV = np.zeros((n_vertices, 2))
    UV[boundary_nodes] = B[boundary_nodes]
    UV[free] = U_free

    return UV


def _rotate_rows(arr, k):
    k = int(k) % len(arr)
    return np.vstack([arr[k:], arr[:k]])


def _best_cyclic_shift(A, B):
    m = len(A)
    best_s, best_val = 0, float('inf')
    for s in range(m):
        Br = np.roll(B, -s, axis=0)
        val = np.sum((A - Br) ** 2)
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
    theta = 2 * np.pi * (np.arange(m) / m)
    circle_positions = np.column_stack([np.cos(theta), np.sin(theta)])
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
    close_inds = np.where(
        np.isclose(
            vertices[:,1] - init_systems.Coords.base_origin[1], 0.0,
            atol=1.0
        )
    )
    close_vertices = vertices[close_inds]
    bottom_right_vertex = close_vertices[np.argmax(close_vertices[:,0])]
    return bottom_right_vertex


def _get_bottom_right_idx(vertices):
    bottom_right_vertex = _get_bottom_right_vertex(vertices)
    bottom_right_idx = np.where(
        np.all(np.isclose(vertices - bottom_right_vertex, 0.0), axis=1)
    )[0][0]
    return bottom_right_idx


def get_mapped_vertices(
        init_vertices, all_polygon_inds, boundary_mask, outer_shape
    ):
    boundary_inds = np.where(boundary_mask)[0]
    boundary_vertices = init_vertices[boundary_inds]

    sorted_boundary_inds = init_systems.sort_counterclockwise(
        boundary_inds, boundary_vertices
    )
    sorted_boundary_inds = np.array(sorted_boundary_inds)

    ccw_boundary_vertices = init_vertices[sorted_boundary_inds]

    bottom_right_idx = _get_bottom_right_idx(ccw_boundary_vertices)
    sorted_boundary_inds = np.roll(
        sorted_boundary_inds, -bottom_right_idx, axis=0
    )

    polygons = []
    for polygon_inds in all_polygon_inds:
        polygon = polygon_inds[polygon_inds != -1][:-2]
        polygons.append(polygon.tolist())

    sorted_inds = init_systems.sort_counterclockwise(
        np.arange(outer_shape.shape[0]), outer_shape
    )
    sorted_inds = np.array(sorted_inds)
    outer_shape = outer_shape[sorted_inds]
    bottom_right_idx = _get_bottom_right_idx(outer_shape)
    sorted_outer_shape = np.roll(outer_shape, -bottom_right_idx, axis=0)
    UV_init, mapped_vertices, edges = _map_to_given_shape(
        init_vertices, polygons, sorted_boundary_inds, sorted_outer_shape
    )

    return mapped_vertices
