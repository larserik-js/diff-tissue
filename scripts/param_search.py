import optuna

from diff_tissue.core import shape_opt
from diff_tissue.app import config, parameters


def objective_f(trial):
    params = parameters.Params()

    params = params.replace(
        n_morph_steps=trial.suggest_int("morph steps", 50, 2000)
    )
    params = params.replace(
        areas_pot_weight=trial.suggest_float(
            "areas_potential_weight", 1.0, 1e4, log=True
        )
    )
    params = params.replace(
        angles_pot_weight=trial.suggest_float(
            "angles_pot_weight", 1.0, 1e4, log=True
        )
    )
    params = params.replace(
        anisotropies_pot_weight=trial.suggest_float(
            "anisotropies_pot_weight", 1.0, 1e4, log=True
        )
    )
    params = params.replace(seed=3)
    params = params.replace(quiet=True)

    sim_states = shape_opt.run(params)
    best = shape_opt.get_best_state(sim_states)
    loss = best.loss

    return loss


def _main():
    cfg = config.load_cfg("config.yml")
    paths = config.ProjectPaths(
        data_base_dir=cfg.data_base_dir,
        outputs_base_dir=cfg.outputs_base_dir,
    )

    db_url = f"sqlite:///{paths.param_search_db}"
    study = optuna.create_study(
        study_name="my_study",
        direction="minimize",
        storage=db_url,
        load_if_exists=True,
    )

    study.optimize(objective_f, n_trials=1000, n_jobs=8)

    print("Best loss:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    _main()
