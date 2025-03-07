import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
import torch


_OPTIMAL_ANGLE =  np.pi / 3
_ANGLES_LOSS_WEIGHT = 0.1
_TARGET_AREA = 2.0
_AREA_VARIANCE_LOSS_WEIGHT = 1e5
_NUM_POLYGONS = 100
_MAX_VERTICES = 15


def _get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def _get_project_dir():
    project_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    return project_dir


def _get_output_dir():
    project_dir = _get_project_dir()
    output_dir = project_dir / 'output'
    return output_dir


def _make_output_dir():
    output_dir = _get_output_dir()
    output_dir.mkdir(exist_ok=True, parents=True)


def _set_seeds():
    np.random.seed(0)
    torch.manual_seed(0)


class _Polygons:
    def __init__(self):
        self._all_polygon_vertex_inds, self._vertices = (
            self._make_init_polygons()
        )
        self._polygon_inds = self._make_polygon_inds()
        self._mask = (self._polygon_inds != -1)

    def _is_finite(self, region):
        return (-1 not in region) and (len(region) > 0)

    def _inside_unit_square(self, vertices):
        inside_unit_square = (
            (vertices >= 0).all(axis=1) & (vertices <= 1).all(axis=1)
        )
        return inside_unit_square

    def _any_vertex_outside_unit_square(self, vertices):
        return np.any((vertices < 0) | (vertices > 1))

    def _extend_region(self, region):
        region.insert(0, region[-1])
        region.append(region[1])
        return region

    def _make_init_polygons(self):
        circumcenters = np.random.rand(_NUM_POLYGONS, 2)
        vor = Voronoi(circumcenters)
        inside_unit_square = self._inside_unit_square(vor.vertices)
        not_allowed_vertex_inds = np.where(~inside_unit_square)[0]
        allowed_vertices = vor.vertices[inside_unit_square]

        all_polygon_vertex_inds = []
        for region in vor.regions:
            if self._is_finite(region):
                vertices = vor.vertices[region]
                if self._any_vertex_outside_unit_square(vertices):
                    continue
            
                # For efficiency
                region = self._extend_region(region)
                vertex_inds = np.array(region)
                adjustment_inds = np.zeros_like(vertex_inds)
                for i in not_allowed_vertex_inds:
                    adjustment_inds -= (vertex_inds >= i).astype(int)
                vertex_inds += adjustment_inds
                all_polygon_vertex_inds.append(vertex_inds)

        return all_polygon_vertex_inds, allowed_vertices

    def _make_polygon_inds(self):
        all_polygon_inds = []
        for vertex_inds in self._all_polygon_vertex_inds:
            n_padding_values = _MAX_VERTICES - len(vertex_inds)
            padding_tensor = torch.full(
                (n_padding_values,), -1, dtype=torch.long
            )
            polygon_inds = torch.cat(
                [torch.tensor(vertex_inds), padding_tensor]
            )
            all_polygon_inds.append(polygon_inds)
    
        all_polygon_inds = torch.stack(all_polygon_inds)
        return all_polygon_inds

    def get_polygon_inds(self):
        return self._polygon_inds
    
    def get_mask(self):
        return self._mask
    
    def get_vertices(self):
        return self._vertices


# def _calc_area(vertices):
#     xs = vertices[1:-1, 0]
#     y_plus_ones = vertices[2:, 1]
#     y_minus_ones = vertices[:-2, 1]
#     first_term = torch.dot(xs, y_plus_ones)
#     second_term = torch.dot(xs, y_minus_ones)
#     # Abs. because vertex orientation can be
#     # both clockwise and counter-clockwise
#     area = 0.5 * torch.abs(first_term - second_term)
#     return area


# def _calc_edges(vertices):
#     return vertices[1:] - vertices[:-1]


# def _calc_angles_loss(vertices):
#     edges = _calc_edges(vertices)
#     dot_products = torch.sum(edges[:-1] * edges[1:], dim=1)
#     norms = torch.norm(edges, dim=1)
#     cosines = torch.clip(dot_products / (norms[:-1] * norms[1:]), -1.0, 1.0)
#     angles = torch.acos(cosines)
#     angles_loss = torch.sum((angles - _OPTIMAL_ANGLE)**2)
#     return angles_loss


def _calc_all_areas(all_cells, mask):
    xs = all_cells[:, 1:-1, 0]
    y_plus_ones = all_cells[:, 2:, 1]
    y_minus_ones = all_cells[:, :-2, 1]

    valid = mask[:, 1:-1] & mask[:, 2:] & mask[:, :-2]

    first_term = xs * y_plus_ones
    first_term = torch.sum(first_term * valid, dim=1)
    second_term = xs * y_minus_ones
    second_term = torch.sum(second_term * valid, dim=1)

    # Abs. because vertex orientation can be
    # both clockwise and counter-clockwise
    areas = 0.5 * torch.abs(first_term - second_term)

    return areas


def _calc_all_angles_loss(all_cells, mask):
    valid = mask[:, 1:] & mask[:, :-1]
    valid = valid[:, 1:] & valid[:, :-1]

    edges = all_cells[:, 1:] - all_cells[:, :-1]
    dot_products = torch.sum(edges[:, :-1] * edges[:, 1:], dim=2)
    norms = torch.norm(edges, dim=2)
    cosines = dot_products / (1e-7 + norms[:, :-1] * norms[:, 1:])
    angles = torch.acos(cosines)
    angles_loss = torch.sum((angles - _OPTIMAL_ANGLE)**2 * valid)

    return angles_loss


def _calc_loss(vertices, indices, mask):
    all_cells = vertices[indices]

    areas = _calc_all_areas(all_cells, mask)

    areas_loss = torch.sum((_TARGET_AREA - areas)**2)
    angles_loss = _ANGLES_LOSS_WEIGHT * _calc_all_angles_loss(all_cells, mask)
    area_variance_loss = _AREA_VARIANCE_LOSS_WEIGHT * torch.var(areas)

    loss = areas_loss + angles_loss + area_variance_loss

    return loss


def _make_figure(vertices, indices, mask):
    fig, ax = plt.subplots()
    range_ = np.array([-2, 2]) + 0.5
    ax.set_xlim(range_)
    ax.set_ylim(range_)

    for i in range(indices.shape[0]):
        vertex_inds = indices[i][mask[i]]
        polygon = vertices[vertex_inds].cpu().detach().numpy()
        ax.scatter(polygon[:, 0], polygon[:, 1], color='green', zorder=1)
        ax.plot(polygon[:, 0], polygon[:, 1], color='black', zorder=2)
    
    return fig


def _save_figure(fig, step):
    output_dir = _get_output_dir()
    fig_path = output_dir / f'step_{step}.pdf'
    
    fig.savefig(fig_path)
    plt.close()


def _main():
    _set_seeds()
    _make_output_dir()
    device = _get_device()

    polygons = _Polygons()
    vertices = torch.tensor(
        polygons.get_vertices(), device=device, requires_grad=True,
        dtype=torch.float32
    )
    indices = torch.tensor(polygons.get_polygon_inds(), device=device)
    mask = torch.tensor(polygons.get_mask(), device=device)

    optimizer = torch.optim.Adam([vertices], lr=0.0005)

    n_steps = 10000
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = _calc_loss(vertices, indices, mask)
        loss.backward()
        optimizer.step()

        if step % int(n_steps / 100) == 0:
            print(f'Step {step}, Loss: {loss.item()}')
        
            fig = _make_figure(vertices, indices, mask)
            _save_figure(fig, step)


if __name__ == "__main__":
    _main()
