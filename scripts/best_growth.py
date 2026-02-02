import jax
import numpy as np
import pandas as pd

from diff_tissue import morphing, my_files, my_utils, parameters


def _save_best_growth_evolution(growth_evolution, params):
    output_file = my_files.OutputFile('best_growth', '.pkl', params)
    data_handler = my_files.DataHandler(output_file)
    data_handler.save(growth_evolution)


def main():
    jax.config.update('jax_enable_x64', True)

    params = parameters.Params().params

    np.random.seed(params.seed)

    jax_arrays = my_utils.get_jax_arrays(params)

    input_file = my_files.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)

    best_goal_areas = my_utils.to_jax(df['best_goal_area'].values)
    best_goal_elongations = my_utils.to_jax(df['best_goal_elongation'].values)

    jiterate = jax.jit(morphing.iterate, static_argnames=['n_steps'])

    growth_evolution = jiterate(
        best_goal_areas, best_goal_elongations, params.n_growth_steps,
        jax_arrays, params
    )

    _save_best_growth_evolution(growth_evolution, params)


if __name__ == '__main__':
    main()
