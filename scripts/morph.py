from diff_tissue.jax_bootstrap import jax, jnp
from diff_tissue import morphing, my_files, my_utils, parameters


def _save_growth_evolution(growth_evolution, params):
    output_file = my_files.OutputFile('morphing', '.pkl', params)
    data_handler = my_files.DataHandler(output_file)
    data_handler.save(growth_evolution)


@my_utils.timer
def _main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    init_areas = jax_arrays['init_areas']

    goal_areas = 2.0 * init_areas
    goal_elongations = 5.0 * jnp.ones_like(init_areas)

    jiterate = jax.jit(morphing.iterate, static_argnames=['n_steps'])

    growth_evolution = jiterate(
        goal_areas, goal_elongations, params.n_growth_steps, jax_arrays, params
    )

    _save_growth_evolution(growth_evolution, params)


if __name__ == '__main__':
    _main()
