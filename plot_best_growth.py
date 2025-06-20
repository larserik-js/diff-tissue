import jax
import numpy as np
import pandas as pd

import growth, my_utils


def main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    best_growth_dir = my_utils.OutputDir('best_growth', params)
    best_growth_dir.make()

    jax_arrays = my_utils.get_jax_arrays(params)

    input_file = my_utils.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)
    
    best_goal_areas = my_utils.to_jax(df['goal_area'].values)
    best_goal_aspect_ratios = my_utils.to_jax(df['goal_aspect_ratio'].values)

    growth.iterate_and_plot(
        best_growth_dir.get_param_path(), best_goal_areas,
        best_goal_aspect_ratios, jax_arrays, params.numerical
    )


if __name__ == '__main__':
    main()
