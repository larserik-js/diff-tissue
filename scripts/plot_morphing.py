import jax
import numpy as np

from diff_tissue import my_files, my_utils, parameters, plotting


def _get_plotting_data(params):
    input_file = my_files.OutputFile('morphing', '.pkl', params)
    data_handler = my_files.DataHandler(input_file)
    growth_evolution = data_handler.load()
    return growth_evolution


def _plot(growth_evolution, output_dir, jax_arrays, params):
    figure = plotting.MorphFigure(output_dir, jax_arrays, params)

    for t, vertices in enumerate(growth_evolution):
        if t%10 == 0:
            figure.save_plot(vertices, t)
    figure.save_plot(vertices, t)


def _main():
    jax.config.update('jax_enable_x64', True)

    params = parameters.Params()

    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    growth_evolution = _get_plotting_data(params)

    output_dir = my_files.OutputDir('morphing', params).path

    _plot(growth_evolution, output_dir, jax_arrays, params)


if __name__ == '__main__':
    _main()
