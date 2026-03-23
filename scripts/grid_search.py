import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from diff_tissue.app import grid_search


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["petal", "trapezoid", "triangle", "nconv"],
        help="List of shapes.",
    )
    parser.add_argument("--arpws", nargs=3, required=True, type=float)
    parser.add_argument("--aspws", nargs=3, required=True, type=float)
    parser.add_argument("--anpws", nargs=3, required=True, type=float)
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
        default=2,
        dest="n_workers",
        help="Number of workers.",
    )
    return parser.parse_args()


@dataclass
class _GridVariables:
    shapes: list[str]
    areas_pot_ws: NDArray[np.floating]
    anisotropies_pot_ws: NDArray[np.floating]
    angles_pot_ws: NDArray[np.floating]


def _parse_arange(values):
    """Convert CLI input into np.arange arguments."""
    if len(values) != 3:
        raise ValueError("Expected exactly 3 values: start stop step")
    start, stop, step = map(float, values)
    return np.arange(start, stop, step)


def _main():
    args = _parse_args()

    areas_pot_ws = _parse_arange(args.arpws)
    anisotropies_pot_ws = _parse_arange(args.aspws)
    angles_pot_ws = _parse_arange(args.anpws)

    grid_variables = _GridVariables(
        shapes=args.shapes,
        areas_pot_ws=areas_pot_ws,
        anisotropies_pot_ws=anisotropies_pot_ws,
        angles_pot_ws=angles_pot_ws,
    )

    grid_search.run(grid_variables, args.study_name, args.n_workers)


if __name__ == "__main__":
    _main()
