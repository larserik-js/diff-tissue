from diff_tissue import io_utils, my_utils, parameters, plotting


def _get_plotting_data(params):
    input_file = io_utils.OutputFile('final_tissues', '.pkl', params)
    data_handler = io_utils.DataHandler(input_file)
    final_tissues = data_handler.load()
    return final_tissues


def _plot(final_tissues, output_dir, jax_arrays, params):
    figure = plotting.MorphFigure(output_dir, jax_arrays, params)

    for t, vertices in enumerate(final_tissues):
        if t%10 == 0:
            figure.save_plot(vertices, t, enumerate=True)
    figure.save_plot(vertices, t, enumerate=True)


def _main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    final_tissues = _get_plotting_data(params)

    output_dir = io_utils.OutputDir('final_tissues', params).path

    _plot(final_tissues, output_dir, jax_arrays, params)


if __name__ == '__main__':
    _main()
