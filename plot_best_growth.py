import jax
import numpy as np
import pandas as pd

import morph, my_files, my_utils, plotting


def _plot(growth_evolution, output_dir, jax_arrays, params):
    figure = plotting.MorphGrowthFigure(output_dir, jax_arrays, params)

    for t, vertices in enumerate(growth_evolution):
        if t%2 == 0:
            figure.save_plot(vertices, step=t)

    # Always plot final state
    figure.save_plot(vertices, step=t)


def main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    jax_arrays = my_utils.get_jax_arrays(params)

    input_file = my_files.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)
    
    best_goal_areas = my_utils.to_jax(df['best_goal_area'].values)
    best_goal_aspect_ratios = my_utils.to_jax(
        df['best_goal_aspect_ratio'].values
    )

    growth_evolution = morph.iterate(
        best_goal_areas, best_goal_aspect_ratios,
        params.numerical['n_growth_steps'], jax_arrays, params.numerical
    )

    output_dir = my_files.OutputDir('best_growth', params).path

    _plot(growth_evolution, output_dir, jax_arrays, params)


if __name__ == '__main__':
    main()
