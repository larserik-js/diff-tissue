import optuna

from diff_tissue.app import io_utils


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


def _run(db_path, study_name):
    db_url = f"sqlite:///{db_path}"
    storage = optuna.storages.RDBStorage(db_url)

    _show_studies(storage)

    study = optuna.load_study(study_name=f"{study_name}", storage=storage)

    _show_first_trials(study)
    _show_best_trial(study)
    _show_n_completed(study)


def inspect_param_search(study_name):
    output_manager = io_utils.OutputManager(None, base_dir="outputs")
    db_path = output_manager.file_path("optuna.db")

    _run(db_path, study_name)
