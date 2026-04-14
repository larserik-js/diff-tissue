import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from diff_tissue.app import config, shape_opt


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--systems",
        nargs="+",
        default=["few"],
        help="List of shapes.",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["petal"],
        help="List of shapes.",
    )
    parser.add_argument("--knots", default=False, action="store_true")
    parser.add_argument("--trans", nargs=3, default=[75.0, 80.0, 5.0])
    parser.add_argument("--msteps", nargs=3, default=[500, 600, 100])
    parser.add_argument("--arpws", nargs=3, default=[5.0, 6.0, 1.0])
    parser.add_argument("--aspws", nargs=3, default=[50.0, 51.0, 1.0])
    parser.add_argument("--anpws", nargs=3, default=[13.0, 14.0, 1.0])
    parser.add_argument("--init_lrs", nargs=3, default=[0.01, 0.02, 0.01])
    parser.add_argument("--ssteps", nargs=3, default=[1000, 2000, 1000])
    parser.add_argument("--slws", nargs=3, default=[1.0, 2.0, 1.0])
    parser.add_argument("--vlws", nargs=3, default=[0.0, 1.0, 1.0])
    parser.add_argument("--ids", nargs=3, default=[0, 1, 1])
    parser.add_argument("--ilws", nargs=3, default=[0.5, 0.6, 0.1])
    parser.add_argument("--seeds", nargs=3, default=[0, 1, 1])
    parser.add_argument(
        "--w",
        type=int,
        default=1,
        dest="n_workers",
        help="Number of workers.",
    )
    return parser.parse_args()


def _parse_arange(values, dtype):
    """Convert CLI input into np.arange arguments."""
    if len(values) != 3:
        raise ValueError("Expected exactly 3 values: start stop step")
    start, stop, step = map(dtype, values)
    return np.arange(start, stop, step, dtype=dtype)


@dataclass
class _GridVariables:
    system: list[str]
    shape: list[str]
    knots: list[bool]
    quiet: list[bool]
    trapezoid_angle: NDArray[np.floating]
    n_morph_steps: NDArray[np.integer]
    areas_pot_weight: NDArray[np.floating]
    anisotropies_pot_weight: NDArray[np.floating]
    angles_pot_weight: NDArray[np.floating]
    init_lr: NDArray[np.floating]
    n_shape_steps: NDArray[np.integer]
    shape_loss_weight: NDArray[np.floating]
    var_loss_weight: NDArray[np.floating]
    poly_id_cfg: NDArray[np.integer]
    poly_id_loss_weight: NDArray[np.floating]
    seed: NDArray[np.integer]


def _main():
    args = _parse_args()

    grid_variables = _GridVariables(
        system=args.systems,
        shape=args.shapes,
        knots=[args.knots],
        quiet=[True],
        trapezoid_angle=_parse_arange(args.trans, dtype=float),
        n_morph_steps=_parse_arange(args.msteps, dtype=int),
        areas_pot_weight=_parse_arange(args.arpws, dtype=float),
        anisotropies_pot_weight=_parse_arange(args.aspws, dtype=float),
        angles_pot_weight=_parse_arange(args.anpws, dtype=float),
        init_lr=_parse_arange(args.init_lrs, dtype=float),
        n_shape_steps=_parse_arange(args.ssteps, dtype=int),
        shape_loss_weight=_parse_arange(args.slws, dtype=float),
        var_loss_weight=_parse_arange(args.vlws, dtype=float),
        poly_id_cfg=_parse_arange(args.ids, dtype=int),
        poly_id_loss_weight=_parse_arange(args.ilws, dtype=float),
        seed=_parse_arange(args.seeds, dtype=int),
    )

    cfg = config.load_cfg("config.yml")
    paths = config.ProjectPaths(cfg.data_base_dir, cfg.outputs_base_dir)

    shape_opt.run_multi(grid_variables, paths, n_workers=args.n_workers)


if __name__ == "__main__":
    _main()
