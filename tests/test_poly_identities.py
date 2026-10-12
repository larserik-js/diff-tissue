from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from diff_tissue.app import parameters
from diff_tissue.core import init_systems, metrics, poly_identities
from diff_tissue.core.jax_bootstrap import jnp


_JaxPolyIdentities = (
    poly_identities._JaxProxDistIdentities
    | poly_identities._JaxMidOuterIdentities
)
_DistFn = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]


def _y_dist_from_base(
    centroids: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return centroids[:, 1] - init_systems.Coords.base_origin[1]


def _x_dist_from_mid_axis(
    centroids: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    return np.abs(centroids[:, 0] - init_systems.Coords.base_origin[0])


def test_no_poly_identities() -> None:
    params = parameters.Params(poly_id_cfg=0)
    assert poly_identities.get_poly_identities(params) is None


@dataclass(frozen=True)
class _IdentityCfg:
    """Splits polygons into an inner group, which the loss acts on, and an
    outer group, by a distance of their initial centroids."""

    poly_id_cfg: int
    inner_attr: str
    outer_attr: str
    dist_fn: _DistFn

    def get_groups(
        self,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        params = parameters.Params(poly_id_cfg=self.poly_id_cfg)
        poly_ids = poly_identities.get_poly_identities(params)
        inner = np.ravel(getattr(poly_ids, self.inner_attr))
        outer = np.ravel(getattr(poly_ids, self.outer_attr))
        return inner, outer


_IDENTITY_CFGS = [
    pytest.param(
        _IdentityCfg(1, "proximal_inds", "distal_inds", _y_dist_from_base),
        id="prox_dist",
    ),
    pytest.param(
        _IdentityCfg(2, "mid_inds", "outer_inds", _x_dist_from_mid_axis),
        id="mid_outer",
    ),
]


@pytest.mark.parametrize("cfg", _IDENTITY_CFGS)
def test_poly_identities_partition_polygons(cfg: _IdentityCfg) -> None:
    n_polygons = len(init_systems.get_system(parameters.Params()).indices)
    inner, outer = cfg.get_groups()

    assert len(inner) > 0 and len(outer) > 0
    assert set(inner).isdisjoint(outer)
    assert len(inner) + len(outer) == n_polygons


@pytest.mark.parametrize("cfg", _IDENTITY_CFGS)
def test_poly_identities_split_by_distance(cfg: _IdentityCfg) -> None:
    polygons = init_systems.get_system(parameters.Params())
    centroids = metrics.calc_centroids(
        polygons.init_vertices, polygons.indices, polygons.valid_mask
    )
    dists = cfg.dist_fn(centroids)
    inner, outer = cfg.get_groups()

    assert np.max(dists[inner]) < np.min(dists[outer])


# Four polygons: the first two are inner, the last two outer
_INNER_INDS = jnp.array([[0], [1]])
_OUTER_INDS = jnp.array([[2], [3]])
_LOSS_CFGS = [
    pytest.param(
        1,
        poly_identities._JaxProxDistIdentities(
            proximal_inds=_INNER_INDS, distal_inds=_OUTER_INDS
        ),
        id="prox_dist",
    ),
    pytest.param(
        2,
        poly_identities._JaxMidOuterIdentities(
            mid_inds=_INNER_INDS, outer_inds=_OUTER_INDS
        ),
        id="mid_outer",
    ),
]


def _calc_loss(
    poly_id_cfg: int,
    poly_ids: _JaxPolyIdentities | None,
    anisotropies: Sequence[float],
) -> float:
    vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    indices = jnp.array([[3, 0, 1, 2, 3, 0]] * len(anisotropies))
    poly_metrics = metrics.initialize_poly_metrics(
        vertices, indices, indices != -1
    )
    poly_metrics = poly_metrics.replace(anisotropies=jnp.array(anisotropies))
    loss = poly_identities.calc_poly_id_loss(
        poly_id_cfg, poly_ids, poly_metrics
    )
    return float(loss)


def test_poly_id_loss_without_identities_is_zero() -> None:
    assert _calc_loss(0, None, [0.0, 0.5, -0.5, 1.0]) == pytest.approx(0.0)


@pytest.mark.parametrize(("poly_id_cfg", "poly_ids"), _LOSS_CFGS)
def test_poly_id_loss_is_zero_for_tall_inner_polygons(
    poly_id_cfg: int, poly_ids: _JaxPolyIdentities
) -> None:
    loss = _calc_loss(poly_id_cfg, poly_ids, [1.0, 1.0, 0.0, -1.0])
    assert loss == pytest.approx(0.0)


@pytest.mark.parametrize(("poly_id_cfg", "poly_ids"), _LOSS_CFGS)
def test_poly_id_loss_increases_away_from_tall(
    poly_id_cfg: int, poly_ids: _JaxPolyIdentities
) -> None:
    losses = [
        _calc_loss(poly_id_cfg, poly_ids, [anisotropy, anisotropy, 0.0, 0.0])
        for anisotropy in [1.0, 0.5, 0.0, -0.5, -1.0]
    ]
    assert np.all(np.diff(losses) > 0.0)


@pytest.mark.parametrize(("poly_id_cfg", "poly_ids"), _LOSS_CFGS)
def test_poly_id_loss_ignores_outer_polygons(
    poly_id_cfg: int, poly_ids: _JaxPolyIdentities
) -> None:
    loss = _calc_loss(poly_id_cfg, poly_ids, [0.5, 0.0, 1.0, 1.0])
    loss_other_outer = _calc_loss(poly_id_cfg, poly_ids, [0.5, 0.0, -1.0, 0.3])
    assert loss == pytest.approx(loss_other_outer)
