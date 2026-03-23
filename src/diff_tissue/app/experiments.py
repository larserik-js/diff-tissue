from ..core import init_systems
from ..core import shape_opt as shape_opt_core
from . import io_utils, learned_morph, morphing, parameters, tutte_fields
from . import shape_opt as shape_opt_app


OUTPUT_DIR = "outputs"


def run_morphing(params, base_dir=OUTPUT_DIR):
    polygons = init_systems.get_jax_polygons(params)

    output = io_utils.OutputManager(
        morphing.OUTPUT_TYPE_DIR, base_dir=base_dir
    )

    param_string = parameters.get_param_string(params)
    cache_path = output.cache_path(f"{param_string}.npz")

    morph_evolution = morphing.get_morph_evolution(
        cache_path, polygons, params
    )

    morphing.save_figs(morph_evolution, output, param_string, params)


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

    cache_path = output.cache_path(f"best_morph__{param_string}.pkl")
    best_morph_evolution = shape_opt_app.get_best_morph_evolution(
        best_goal_areas, best_goal_anisotropies, polygons, params, cache_path
    )

    shape_opt_app.plot_best_morph(
        best_morph_evolution, output, param_string, params
    )


def run_learned_morph(params):
    output = io_utils.OutputManager(
        learned_morph.OUTPUT_TYPE_DIR, base_dir=OUTPUT_DIR
    )

    results = learned_morph.run(params, output)

    learned_morph.plot(results, params, output)


def plot_tutte_fields():
    output = io_utils.OutputManager(
        f"{tutte_fields.OUTPUT_TYPE_DIR}", base_dir="outputs"
    )

    shape = "petal"

    tutte_fields_ = tutte_fields.get_fields(shape, output)

    fig = tutte_fields.plot(tutte_fields_)

    tutte_fields.save_plot(fig, shape, output)
