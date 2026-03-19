import argparse

import numpy as np

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
    return parser.parse_args()


def _main():
    args = _parse_args()

    shapes = ["petal", "trapezoid", "triangle", "nconv"]
    areas_pot_ws = np.arange(1.0, 50.0, 4)
    anisotropies_pot_ws = np.arange(1.0, 50.0, 4)
    angles_pot_ws = np.arange(1.0, 50.0, 4)

    grid_search.run(
        args.study_name,
        shapes,
        areas_pot_ws,
        anisotropies_pot_ws,
        angles_pot_ws,
    )


if __name__ == "__main__":
    _main()
