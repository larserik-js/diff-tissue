from functools import cached_property
from typing import Union

import numpy as np

from .jax_bootstrap import jnp, struct
from . import init_systems, metrics


class _ProxDistIdentities:
    def __init__(self, tutte_centroids):
        self._tutte_centroids = tutte_centroids
        self._proximal_dist = 4.0  # From paper

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


def _calc_prox_dist_loss(poly_ids, poly_metrics):
    proximal_anisotropies = poly_metrics.anisotropies[poly_ids.proximal_inds]
    anisotropy_loss = jnp.mean(jnp.square(proximal_anisotropies - 1.0))
    return anisotropy_loss


@struct.dataclass
class _JaxProxDistIdentities:
    proximal_inds: jnp.ndarray
    distal_inds: jnp.ndarray


class _MidOuterIdentities:
    def __init__(self, tutte_centroids):
        self._tutte_centroids = tutte_centroids
        self._dist_from_mid_axis = 3.0  # From paper

    @cached_property
    def _x_dists_from_mid_axis(self):
        x_dists_from_mid_axis = np.abs(
            self._tutte_centroids[:, 0] - init_systems.Coords.base_origin[0]
        )
        return x_dists_from_mid_axis

    @cached_property
    def mid_inds(self):
        mid_inds_ = np.argwhere(
            self._x_dists_from_mid_axis <= self._dist_from_mid_axis
        )
        return mid_inds_

    @cached_property
    def outer_inds(self):
        outer_inds_ = np.argwhere(
            self._x_dists_from_mid_axis > self._dist_from_mid_axis
        )
        return outer_inds_


def _calc_mid_outer_loss(poly_ids, poly_metrics):
    mid_anisotropies = poly_metrics.anisotropies[poly_ids.mid_inds]
    anisotropy_loss = jnp.mean(jnp.square(mid_anisotropies - 1.0))
    return anisotropy_loss


@struct.dataclass
class _JaxMidOuterIdentities:
    mid_inds: jnp.ndarray
    outer_inds: jnp.ndarray


def calc_poly_id_loss(id, poly_ids, poly_metrics):
    if id == 0:
        poly_id_loss = 0.0
    elif id == 1:
        poly_id_loss = 5.0 * _calc_prox_dist_loss(poly_ids, poly_metrics)
    elif id == 2:
        poly_id_loss = 12.5 * _calc_mid_outer_loss(poly_ids, poly_metrics)
    return poly_id_loss


def get_poly_identities(params):
    if params.poly_id_cfg == 0:
        return None
    else:
        polygons = init_systems.get_system(params)
        init_centroids = metrics.calc_centroids(
            polygons.init_vertices, polygons.indices, polygons.valid_mask
        )
        poly_identities: Union[_ProxDistIdentities, _MidOuterIdentities]
        jax_poly_identities: Union[
            _JaxProxDistIdentities, _JaxMidOuterIdentities
        ]
        if params.poly_id_cfg == 1:
            poly_identities = _ProxDistIdentities(init_centroids)
            jax_poly_identities = _JaxProxDistIdentities(
                proximal_inds=jnp.array(poly_identities.proximal_inds),
                distal_inds=jnp.array(poly_identities.distal_inds),
            )
        elif params.poly_id_cfg == 2:
            poly_identities = _MidOuterIdentities(init_centroids)
            jax_poly_identities = _JaxMidOuterIdentities(
                mid_inds=jnp.array(poly_identities.mid_inds),
                outer_inds=jnp.array(poly_identities.outer_inds),
            )

        return jax_poly_identities
