from ..core import init_systems
from ..core import shape_opt as shape_opt_core
from . import io_utils, learned_growth, morphing, parameters, tutte_fields
from . import shape_opt as shape_opt_app


OUTPUT_DIR = "outputs"


def run_morphing(params, base_dir=OUTPUT_DIR):
    polygons = init_systems.get_jax_polygons(params)

    output = io_utils.OutputManager(
        morphing.OUTPUT_TYPE_DIR, base_dir=base_dir
    )

    param_string = parameters.get_param_string(params)
    cache_path = output.cache_path(f"{param_string}.pkl")

    growth_evolution = morphing.get_growth_evolution(
        cache_path, polygons, params
    )

    morphing.save_figs(growth_evolution, output, param_string, params)


def run_shape_opt(params, base_dir=OUTPUT_DIR):
    output = io_utils.OutputManager(
        shape_opt_app.OUTPUT_TYPE_DIR, base_dir=base_dir
    )
    sim_states = shape_opt_app.get_sim_states(params, output)

    shape_opt_app.plot_final_tissues(sim_states.final_vertices, output, params)

    best_state = shape_opt_core.get_best_state(sim_states)
    best_goal_areas = best_state.goal_areas
    best_goal_anisotropies = best_state.goal_anisotropies

    polygons = init_systems.get_jax_polygons(params)
    param_string = parameters.get_param_string(params)

    cache_path = output.cache_path(f"best_growth__{param_string}.pkl")
    best_growth_evolution = shape_opt_app.get_best_growth_evolution(
        best_goal_areas, best_goal_anisotropies, polygons, params, cache_path
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


def plot_tutte_fields():
    output = io_utils.OutputManager(
        f"{tutte_fields.OUTPUT_TYPE_DIR}", base_dir="outputs"
    )

    shape = "petal"

    tutte_fields_ = tutte_fields.get_fields(shape, output)

    fig = tutte_fields.plot(tutte_fields_)

    tutte_fields.save_plot(fig, shape, output)
