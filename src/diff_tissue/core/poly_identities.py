from functools import cached_property

import numpy as np

from .jax_bootstrap import jnp, struct
from . import init_systems, metrics


class _PolyIdentities:
    def __init__(self, tutte_centroids):
        self._tutte_centroids = tutte_centroids
        self._proximal_dist = 4.5  # From paper

    @cached_property
    def _y_dists_from_base(self):
        y_dists_from_base = (
            self._tutte_centroids[:, 1] - init_systems.Coords.base_origin[1]
        )
        return y_dists_from_base

    @cached_property
    def proximal_inds(self):
        proximal_inds_ = np.argwhere(
            self._y_dists_from_base <= self._proximal_dist
        )
        return proximal_inds_

    @cached_property
    def distal_inds(self):
        distal_inds = np.argwhere(
            self._y_dists_from_base > self._proximal_dist
        )
        return distal_inds


@struct.dataclass
class _JaxPolyIdentities:
    proximal_inds: jnp.ndarray
    distal_inds: jnp.ndarray


def get_poly_identities(params):
    if params.poly_id_configuration == 0:
        return None
    elif params.poly_id_configuration == 1:
        polygons = init_systems.get_system(params)
        init_centroids = metrics.calc_centroids(
            polygons.init_vertices, polygons.indices, polygons.valid_mask
        )
        poly_identities = _PolyIdentities(init_centroids)
        jax_poly_identities = _JaxPolyIdentities(
            proximal_inds=jnp.array(poly_identities.proximal_inds),
            distal_inds=jnp.array(poly_identities.distal_inds),
        )
        return jax_poly_identities
