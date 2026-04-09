import argparse
from dataclasses import dataclass
import os

import numpy as np
from numpy.typing import NDArray

from diff_tissue.app import config, grid_search


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["trapezoid"],
        help="List of shapes.",
    )
    parser.add_argument("--knots", default=False, action="store_true")
    parser.add_argument("--trans", nargs=3, default=[80.0, 90.0, 10.0])
    parser.add_argument("--arpws", nargs=3, default=[100.0, 150.0, 50.0])
    parser.add_argument("--aspws", nargs=3, default=[100.0, 150.0, 50.0])
    parser.add_argument("--anpws", nargs=3, default=[100.0, 150.0, 50.0])
    parser.add_argument("--seeds", nargs=3, default=[0, 1, 1])
    parser.add_argument(
        "--n",
        type=str,
        default="base_model",
        dest="study_name",
        help="Study name.",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=os.cpu_count(),
        dest="n_workers",
        help="Number of workers.",
    )
    return parser.parse_args()


@dataclass
class _GridVariables:
    shapes: list[str]
    knots: list[bool]
    trapezoid_angles: NDArray[np.floating]
    areas_pot_ws: NDArray[np.floating]
    anisotropies_pot_ws: NDArray[np.floating]
    angles_pot_ws: NDArray[np.floating]
    seeds: NDArray[np.integer]


def _parse_arange(values, dtype):
    """Convert CLI input into np.arange arguments."""
    if len(values) != 3:
        raise ValueError("Expected exactly 3 values: start stop step")
    start, stop, step = map(dtype, values)
    return np.arange(start, stop, step)


def _main():
    args = _parse_args()

    knots = [args.knots]
    trapezoid_angles = _parse_arange(args.trans, dtype=float)
    areas_pot_ws = _parse_arange(args.arpws, dtype=float)
    anisotropies_pot_ws = _parse_arange(args.aspws, dtype=float)
    angles_pot_ws = _parse_arange(args.anpws, dtype=float)
    seeds = _parse_arange(args.seeds, dtype=int)

    grid_variables = _GridVariables(
        shapes=args.shapes,
        knots=knots,
        trapezoid_angles=trapezoid_angles,
        areas_pot_ws=areas_pot_ws,
        anisotropies_pot_ws=anisotropies_pot_ws,
        angles_pot_ws=angles_pot_ws,
        seeds=seeds,
    )

    paths = config.ProjectPaths(
        data_base_dir=config.load_cfg("config.yml").data_base_dir,
        outputs_base_dir=config.load_cfg("config.yml").outputs_base_dir,
    )
    grid_search.run(
        grid_variables,
        args.study_name,
        args.n_workers,
        paths=paths,
    )


if __name__ == "__main__":
    _main()
