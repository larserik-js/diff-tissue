import jax
import numpy as np

import my_files, my_utils


def _get_plotting_data(params):
    input_file = my_files.OutputFile('growth', '.pkl', params)
    data_handler = my_files.DataHandler(input_file)
    growth_evolution = data_handler.load()
    return growth_evolution


def plot(growth_evolution, output_dir, params):
    jax_arrays = my_utils.get_jax_arrays(params)
    figure = my_utils.Figure(growth_evolution[0])

    for t, vertices in enumerate(growth_evolution):
        if t%10 == 0:
            figure.plot(output_dir, vertices, jax_arrays, step=t)


def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    growth_evolution = _get_plotting_data(params)

    output_dir = my_files.OutputDir('growth', params).path

    plot(growth_evolution, output_dir, params)


if __name__ == '__main__':
    _main()
