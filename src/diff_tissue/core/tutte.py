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


def _mean_value_weights(vertices, edges, eps=1e-12):
    """
    Symmetric mean value (Floater) weights per undirected edge.
    Builds directed MVC weights around each vertex using angularly-sorted
    neighbors, then symmetrizes: w_ij = 0.5*(w_ij_dir + w_ji_dir).
    """
    n = len(vertices)
    nbrs = [[] for _ in range(n)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)

    W_dir = {}
    for i in range(n):
        Ni = nbrs[i]
        m = len(Ni)
        if m < 2:
            continue

        vi = vertices[i]
        vecs = vertices[Ni] - vi
        ang = np.arctan2(vecs[:, 1], vecs[:, 0])
        order = np.argsort(ang)

        Ni = [Ni[k] for k in order]
        vecs = vecs[order]

        lens = np.linalg.norm(vecs, axis=1)
        lens = np.maximum(lens, eps)

        u = vecs / lens[:, None]
        u_prev = np.roll(u, 1, axis=0)
        u_next = np.roll(u, -1, axis=0)

        dots_prev = np.clip(np.sum(u_prev * u, axis=1), -1.0, 1.0)
        dots_next = np.clip(np.sum(u * u_next, axis=1), -1.0, 1.0)

        alpha_prev = np.arccos(dots_prev)
        alpha_next = np.arccos(dots_next)

        w = (np.tan(alpha_prev / 2.0) + np.tan(alpha_next / 2.0)) / lens

        for j, wij in zip(Ni, w):
            if np.isfinite(wij) and wij > 0:
                W_dir[(i, j)] = float(wij)

    W = {}
    for (i, j), wij in W_dir.items():
        wji = W_dir.get((j, i), 0.0)
        w = 0.5 * (wij + wji)
        if w > 0:
            a, b = (i, j) if i < j else (j, i)
            W[(a, b)] = w

    # Fallback to uniform weights for any edge that got no MVC weight
    for i, j in edges:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in W:
            W[(a, b)] = 1.0

    return W


def _tutte_embedding(init_vertices, edges, boundary_inds, boundary_positions):
    n_vertices = len(init_vertices)
    free = np.setdiff1d(np.arange(n_vertices), boundary_inds)

    W = _mean_value_weights(init_vertices, edges)

    L = scipy.sparse.lil_matrix((n_vertices, n_vertices))
    for (i, j), w in W.items():
        L[i, i] += w
        L[j, j] += w
        L[i, j] -= w
        L[j, i] -= w
    L = L.tocsr()

    B = np.zeros((n_vertices, 2))
    B[boundary_inds] = boundary_positions

    L_ff = L[free][:, free]
    L_fb = L[free][:, boundary_inds]
    rhs = -L_fb @ B[boundary_inds]

    U_free = scipy.sparse.linalg.spsolve(L_ff, rhs)

    UV = np.zeros((n_vertices, 2))
    UV[boundary_inds] = B[boundary_inds]
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


def _map_to_given_shape(
        init_vertices, polygons, aligned_boundary_inds, aligned_outer_shape,
        boundary_offset=0, auto_align=False
    ):
    # validate sizes
    m = len(aligned_boundary_inds)
    if len(aligned_outer_shape) != m:
        raise ValueError(
            f'boundary_target length {len(aligned_outer_shape)} != '
            + f'len(boundary_nodes) {m}'
        )

    edges = _cells_to_edges(polygons)

    # Step 1: embed initial to unit circle
    theta = 2 * np.pi * (np.arange(m) / m)
    circle_positions = np.column_stack([np.cos(theta), np.sin(theta)])
    UV_init = _tutte_embedding(
        init_vertices, edges, aligned_boundary_inds, circle_positions
    )

    # Step 2: rotate / align target boundary
    target_positions = _rotate_rows(aligned_outer_shape, boundary_offset)
    if auto_align:
        s = _best_cyclic_shift(circle_positions, target_positions)
        target_positions = _rotate_rows(
            aligned_outer_shape, boundary_offset + s
        )

    # Step 3: embed target shape
    mapped_positions = _tutte_embedding(
        init_vertices, edges, aligned_boundary_inds, target_positions
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


def _align_inds(ccw_vertices, ccw_boundary_inds):
    bottom_right_idx = _get_bottom_right_idx(ccw_vertices)
    aligned_inds = np.roll(
        ccw_boundary_inds, -bottom_right_idx, axis=0
    )
    return aligned_inds


def get_mapped_vertices(
        init_vertices, all_polygon_inds, boundary_mask, outer_shape
    ):
    ccw_boundary_inds = init_systems.get_ccw_boundary_inds(
        init_vertices, boundary_mask
    )
    ccw_boundary_vertices = init_vertices[ccw_boundary_inds]

    aligned_boundary_inds = _align_inds(
        ccw_boundary_vertices, ccw_boundary_inds
    )

    polygons = []
    for polygon_inds in all_polygon_inds:
        polygon = polygon_inds[polygon_inds != -1][:-2]
        polygons.append(polygon.tolist())

    ccw_outer_shape_inds = init_systems.sort_counterclockwise(
        np.arange(outer_shape.shape[0]), outer_shape
    )

    ccw_outer_shape_inds = np.array(ccw_outer_shape_inds)
    ccw_outer_shape = outer_shape[ccw_outer_shape_inds]

    aligned_outer_shape_inds = _align_inds(
        ccw_outer_shape, ccw_outer_shape_inds
    )
    aligned_outer_shape = outer_shape[aligned_outer_shape_inds]

    _, mapped_vertices, _ = _map_to_given_shape(
        init_vertices, polygons, aligned_boundary_inds, aligned_outer_shape
    )

    return mapped_vertices
