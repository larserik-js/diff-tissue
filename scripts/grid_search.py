import numpy as np

from diff_tissue.app import grid_search


def _main():
    shapes = ["petal", "trapezoid", "triangle", "nconv"]
    areas_pot_ws = np.arange(1.0, 50.0, 4)
    anisotropies_pot_ws = np.arange(30.0, 70.0, 4)
    angles_pot_ws = np.arange(1.0, 50.0, 4)

    grid_search.run(shapes, areas_pot_ws, anisotropies_pot_ws, angles_pot_ws)


if __name__ == "__main__":
    _main()
