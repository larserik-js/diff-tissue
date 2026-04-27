from ..core import init_systems
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
    param_string = parameters.get_param_string(params)
    shape_opt_paths = shape_opt_app.ShapeOptPaths(paths, param_string)

    sim_states = shape_opt_app.get_sim_states(
        params, shape_opt_paths.sim_states_data_path
    )

    shape_opt_app.plot_final_tissues(
        sim_states.final_vertices, params, shape_opt_paths.final_tissues_dir
    )

    best_morph_evolution = shape_opt_app.get_best_morph_evolution(
        sim_states,
        params,
        shape_opt_paths.best_morph_data_path,
    )

    shape_opt_app.plot_best_morph(
        best_morph_evolution, params, shape_opt_paths.best_morph_figs_dir
    )


def run_learned_morph(params, project_paths):
    param_string = parameters.get_param_string(params)
    learned_morph_paths = learned_morph.LearnedMorphPaths(
        project_paths, param_string
    )

    results = learned_morph.run(params, learned_morph_paths)

    learned_morph.plot(results, learned_morph_paths.figs_dir)


def plot_tutte_fields(paths):
    shape = "petal"
    tutte_fields_paths = tutte_fields.TutteFieldsPaths(paths)

    tutte_fields_ = tutte_fields.get_fields(shape, tutte_fields_paths)

    fig = tutte_fields.plot(tutte_fields_)

    tutte_fields.save_plot(fig, shape, tutte_fields_paths.output_dir)
