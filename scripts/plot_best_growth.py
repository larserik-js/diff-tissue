from diff_tissue import my_files, my_utils, parameters, plotting


def _get_plotting_data(params):
    input_file = my_files.OutputFile('best_growth', '.pkl', params)
    data_handler = my_files.DataHandler(input_file)
    best_growth_evolution = data_handler.load()
    return best_growth_evolution


def _plot(growth_evolution, output_dir, jax_arrays, params):
    figure = plotting.MorphGrowthFigure(output_dir, jax_arrays, params)

    for t, vertices in enumerate(growth_evolution):
        if t%2 == 0:
            figure.save_plot(vertices, step=t)

    # Always plot final state
    figure.save_plot(vertices, step=t)


def main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    best_growth_evolution = _get_plotting_data(params)

    output_dir = my_files.OutputDir('best_growth', params).path

    _plot(best_growth_evolution, output_dir, jax_arrays, params)


if __name__ == '__main__':
    main()
