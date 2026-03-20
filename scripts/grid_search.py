import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from diff_tissue.app import grid_search


def _parse_args():
    parser = argparse.ArgumentParser()

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


def _main():
    args = _parse_args()

    shapes = ["petal", "trapezoid", "triangle", "nconv"]
    areas_pot_ws = np.arange(1.0, 50.0, 4)
    anisotropies_pot_ws = np.arange(1.0, 50.0, 4)
    angles_pot_ws = np.arange(1.0, 50.0, 4)

    grid_variables = _GridVariables(
        shapes=shapes,
        areas_pot_ws=areas_pot_ws,
        anisotropies_pot_ws=anisotropies_pot_ws,
        angles_pot_ws=angles_pot_ws,
    )

    grid_search.run(grid_variables, args.study_name, args.n_workers)


if __name__ == "__main__":
    _main()
