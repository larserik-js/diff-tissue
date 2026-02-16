from diff_tissue.core import my_utils
from diff_tissue.app import io_utils, parameters, plotting


def _get_plotting_data(params):
    input_path = io_utils.OutputFile('best_growth', '.pkl', params).path
    best_growth_evolution = io_utils.load_pkl(input_path)
    return best_growth_evolution


def _plot(growth_evolution, output_dir, jax_arrays, params):
    figure = plotting.MorphGrowthFigure(jax_arrays, params)

    for t, vertices in enumerate(growth_evolution):
        if t%2 == 0:
            fig_path = output_dir / f'step={t:03d}.png'
            figure.save_plot(vertices, step=t, fig_path=fig_path)

    # Always plot final state
    fig_path = output_dir / f'step={t:03d}.png'
    figure.save_plot(vertices, step=t, fig_path=fig_path)


def main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    best_growth_evolution = _get_plotting_data(params)

    output_dir = io_utils.OutputDir('best_growth', params).path

    _plot(best_growth_evolution, output_dir, jax_arrays, params)


if __name__ == '__main__':
    main()
