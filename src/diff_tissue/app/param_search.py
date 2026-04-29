from pathlib import Path

import optuna

from . import io_utils, parameters
from diff_tissue.core import shape_opt


class ParamSearchPaths:
    def __init__(self, project_paths):
        self._project_paths = project_paths

    @property
    def param_search_db(self):
        db_path = Path(self._project_paths.interim_data_dir, "optuna.db")
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
    io_utils.ensure_parent_dir(param_search_paths.param_search_db)

    study = optuna.create_study(
        study_name="my_study",
        direction="minimize",
        storage=param_search_paths.db_url,
        load_if_exists=True,
    )

    study.optimize(_objective_f, n_trials=1000, n_jobs=8)

    print("Best loss:", study.best_value)
    print("Best params:", study.best_params)


def _show_studies(storage):
    study_summaries = optuna.get_all_study_summaries(storage)

    for s in study_summaries:
        print(
            f"Study name: {s.study_name}, "
            f"Direction: {s.direction}, "
            f"Trials: {s.n_trials}, "
            f"Best value: {s.best_trial.value if s.best_trial else None}"
        )
    print("")


def _show_first_trials(study):
    df = study.trials_dataframe()
    print(df.head())
    print("")


def _show_best_trial(study):
    print(f"{'Best loss:':<30}{study.best_value}")
    print("")

    for param, val in study.best_params.items():
        print(f"{param + ':':<30}{val}")
    print("")


def _show_n_completed(study):
    trials = study.get_trials(deepcopy=False)
    n_completed = sum(1 for t in trials if t.state.is_finished())
    print(f"Trials completed: {n_completed}")


def inspect_param_search(paths, study_name):
    param_search_paths = ParamSearchPaths(paths)

    if not param_search_paths.param_search_db.exists():
        print("No param search database found.")
    else:
        storage = optuna.storages.RDBStorage(param_search_paths.db_url)

        _show_studies(storage)

        study = optuna.load_study(study_name=f"{study_name}", storage=storage)

        _show_first_trials(study)
        _show_best_trial(study)
        _show_n_completed(study)
