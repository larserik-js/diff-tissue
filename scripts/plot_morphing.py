from diff_tissue import io_utils, my_utils, parameters, plotting


def _get_plotting_data(params):
    input_path = io_utils.OutputFile('morphing', '.pkl', params)
    growth_evolution = io_utils.load_pkl(input_path)
    return growth_evolution


def _plot(growth_evolution, output_dir, jax_arrays, params):
    figure = plotting.MorphFigure(output_dir, jax_arrays, params)

    for t, vertices in enumerate(growth_evolution):
        if t%10 == 0:
            figure.save_plot(vertices, t)
    figure.save_plot(vertices, t)


def _main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    growth_evolution = _get_plotting_data(params)

    output_dir = io_utils.OutputDir('morphing', params).path

    _plot(growth_evolution, output_dir, jax_arrays, params)


if __name__ == '__main__':
    _main()
