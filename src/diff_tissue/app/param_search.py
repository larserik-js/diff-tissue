import optuna

from . import parameters
from diff_tissue.core import shape_opt


class ParamSearchPaths:
    def __init__(self, project_paths):
        self._project_paths = project_paths

    @property
    def param_search_db(self):
        data_dir = self._project_paths.make_subdir(
            self._project_paths.interim_data_dir
        )
        db_path = data_dir / "optuna.db"
        return db_path

    @property
    def db_url(self):
        return f"sqlite:///{self.param_search_db}"


def _objective_f(trial):
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


def run(paths):
    param_search_paths = ParamSearchPaths(paths)

    study = optuna.create_study(
        study_name="my_study",
        direction="minimize",
        storage=param_search_paths.db_url,
        load_if_exists=True,
    )

    study.optimize(_objective_f, n_trials=1000, n_jobs=8)

    print("Best loss:", study.best_value)
    print("Best params:", study.best_params)
