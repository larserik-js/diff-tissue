import pandas as pd

from diff_tissue.jax_bootstrap import jax
from diff_tissue import morphing, io_utils, my_utils, parameters


def _save_best_growth_evolution(growth_evolution, params):
    output_file = io_utils.OutputFile('best_growth', '.pkl', params)
    data_handler = io_utils.DataHandler(output_file)
    data_handler.save(growth_evolution)


def main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    input_file = io_utils.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)

    best_goal_areas = my_utils.to_jax(df['best_goal_area'].values)
    best_goal_anisotropies = my_utils.to_jax(df['best_goal_anisotropy'].values)

    jiterate = jax.jit(morphing.iterate, static_argnames=['n_steps'])

    growth_evolution = jiterate(
        best_goal_areas, best_goal_anisotropies, params.n_growth_steps,
        jax_arrays, params
    )

    _save_best_growth_evolution(growth_evolution, params)


if __name__ == '__main__':
    main()
