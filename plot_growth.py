import jax
import numpy as np

import my_files, my_utils


def _get_plotting_data(params):
    input_file = my_files.OutputFile('growth', '.pkl', params)
    data_handler = my_files.DataHandler(input_file)
    growth_evolution = data_handler.load()
    return growth_evolution


def _plot(growth_evolution, output_dir, jax_arrays):
    figure = my_utils.MorphFigure(output_dir, jax_arrays)

    for t, vertices in enumerate(growth_evolution):
        if t%10 == 0:
            figure.save_plot(vertices, t)
    figure.save_plot(vertices, t)


def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    growth_evolution = _get_plotting_data(params)

    output_dir = my_files.OutputDir('growth', params).path

    _plot(growth_evolution, output_dir, jax_arrays)


if __name__ == '__main__':
    _main()
