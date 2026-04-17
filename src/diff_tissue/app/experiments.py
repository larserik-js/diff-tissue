from ..core import init_systems
from ..core import shape_opt as shape_opt_core
from . import learned_morph, morphing, parameters, tutte_fields
from . import shape_opt as shape_opt_app


def run_morphing(params, paths):
    param_string = parameters.get_param_string(params)
    morphing_paths = morphing.MorphingPaths(paths, param_string)

    polygons = init_systems.get_jax_polygons(params)

    morph_evolution = morphing.get_morph_evolution(
        polygons, params, morphing_paths.data_path
    )

    morphing.save_figs(morph_evolution, params, morphing_paths.output_dir)


def run_shape_opt(params, paths):
    sim_states = shape_opt_app.get_sim_states(params, paths)

    shape_opt_paths = shape_opt_app.ShapeOptPaths(paths)

    param_string = parameters.get_param_string(params)
    output_dir = paths.make_subdir(
        shape_opt_paths.final_tissues_dir,
        param_string,
    )
    shape_opt_app.plot_final_tissues(
        sim_states.final_vertices, params, output_dir
    )

    best_state = shape_opt_core.get_best_state(sim_states)
    best_goal_areas = best_state.goal_areas
    best_goal_anisotropies = best_state.goal_anisotropies

    polygons = init_systems.get_jax_polygons(params)

    data_dir = paths.make_subdir(shape_opt_paths.best_morph_data_dir)
    data_path = data_dir / f"{param_string}.pkl"
    best_morph_evolution = shape_opt_app.get_best_morph_evolution(
        best_goal_areas, best_goal_anisotropies, polygons, params, data_path
    )

    output_dir = paths.make_subdir(
        shape_opt_paths.best_morph_figs_dir,
        param_string,
    )
    shape_opt_app.plot_best_morph(best_morph_evolution, params, output_dir)


def run_learned_morph(params, paths):
    results = learned_morph.run(params, paths)

    param_string = parameters.get_param_string(params)
    output_dir = paths.make_subdir(
        paths.outputs_base_dir, learned_morph.OUTPUT_TYPE_DIR, param_string
    )
    learned_morph.plot(results, output_dir)


def plot_tutte_fields(paths):
    shape = "petal"
    tutte_fields_paths = tutte_fields.TutteFieldsPaths(paths)

    tutte_fields_ = tutte_fields.get_fields(shape, tutte_fields_paths)

    fig = tutte_fields.plot(tutte_fields_)

    tutte_fields.save_plot(fig, shape, tutte_fields_paths.output_dir)
