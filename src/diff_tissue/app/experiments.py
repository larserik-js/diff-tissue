from ..core import my_utils
from . import io_utils, learned_growth, morphing, parameters
from . import shape_opt as shape_opt_app


OUTPUT_DIR = "outputs"


def run_morphing(params, base_dir=OUTPUT_DIR):
    jax_arrays = my_utils.get_jax_arrays(params)

    output = io_utils.OutputManager(
        morphing.OUTPUT_TYPE_DIR, base_dir=base_dir
    )

    param_string = parameters.get_param_string(params)
    cache_path = output.cache_path(f"{param_string}.pkl")

    growth_evolution = morphing.get_growth_evolution(
        cache_path, jax_arrays, params
    )

    morphing.save_figs(growth_evolution, output, param_string, params)


def run_shape_opt(params, base_dir=OUTPUT_DIR):
    output = io_utils.OutputManager(
        shape_opt_app.OUTPUT_TYPE_DIR, base_dir=base_dir
    )

    shape_opt_app.plot_final_tissues(params, output)

    best_goal_areas, best_goal_anisotropies = shape_opt_app.load_output_params(
        params
    )

    jax_arrays = my_utils.get_jax_arrays(params)
    param_string = parameters.get_param_string(params)

    cache_path = output.cache_path(f"best_growth__{param_string}.pkl")
    best_growth_evolution = shape_opt_app.get_best_growth_evolution(
        best_goal_areas, best_goal_anisotropies, jax_arrays, params, cache_path
    )

    shape_opt_app.plot_best_growth(
        best_growth_evolution, output, param_string, params
    )


def run_learned_growth(params):
    output = io_utils.OutputManager(
        learned_growth.OUTPUT_TYPE_DIR, base_dir=OUTPUT_DIR
    )

    results = learned_growth.run(params, output)

    learned_growth.plot(results, params, output)
