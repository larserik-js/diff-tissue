from ..core import my_utils
from ..core import shape_opt as shape_opt_core
from . import io_utils, morphing, parameters
from . import shape_opt as shape_opt_app


def run_morphing(params, base_dir='outputs'):
    jax_arrays = my_utils.get_jax_arrays(params)

    output = io_utils.OutputManager(morphing.OUTPUT_TYPE_DIR, base_dir=base_dir)

    param_string = parameters.get_param_string(params)
    cache_path = output.cache_path(f'{param_string}.pkl')

    growth_evolution = morphing.get_growth_evolution(
        cache_path, jax_arrays, params
    )

    morphing.save_figs(
        growth_evolution, output, param_string, jax_arrays, params
    )


def run_shape_opt(params, base_dir='outputs'):
    jax_arrays = my_utils.get_jax_arrays(params)
    param_string = parameters.get_param_string(params)

    _, final_tissues, _, tabular_output = shape_opt_core.run(params)

    shape_opt_app.save_output_params(tabular_output, params)

    output = io_utils.OutputManager(
        shape_opt_app.FINAL_TISSUES_DIR, base_dir=base_dir
    )

    cache_path = output.cache_path(f'{param_string}.pkl')

    io_utils.save_pkl(cache_path, final_tissues)

    final_tissues = io_utils.load_pkl(cache_path)

    shape_opt_app.plot_final_tissues(
        final_tissues, output, param_string, jax_arrays, params
    )

    # Best growth
    output = io_utils.OutputManager(
        shape_opt_app.BEST_GROWTH_DIR, base_dir=base_dir
    )
    best_goal_areas, best_goal_anisotropies = shape_opt_app.load_output_params(
        params
    )

    growth_evolution = morphing.jiterate(
        best_goal_areas, best_goal_anisotropies, params.n_growth_steps,
        jax_arrays, params
    )

    cache_path = output.cache_path(f'{param_string}.pkl')
    io_utils.save_pkl(cache_path, growth_evolution)

    best_growth_evolution = io_utils.load_pkl(cache_path)

    shape_opt_app.plot_best_growth(
        best_growth_evolution, output, param_string, jax_arrays, params
    )
