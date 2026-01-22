import jax
import numpy as np

from diff_tissue import my_files, my_utils, plotting


def _get_plotting_data(params):
    input_file = my_files.OutputFile('final_tissues', '.pkl', params)
    data_handler = my_files.DataHandler(input_file)
    final_tissues = data_handler.load()
    return final_tissues


def _plot(final_tissues, output_dir, jax_arrays, params):
    figure = plotting.MorphFigure(output_dir, jax_arrays, params)

    for t, vertices in enumerate(final_tissues):
        if t%10 == 0:
            figure.save_plot(vertices, t, enumerate=True)
    figure.save_plot(vertices, t, enumerate=True)


def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    final_tissues = _get_plotting_data(params)

    output_dir = my_files.OutputDir('final_tissues', params).path

    _plot(final_tissues, output_dir, jax_arrays, params)


if __name__ == '__main__':
    _main()
