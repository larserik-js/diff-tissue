import argparse

import optuna

from diff_tissue.app import io_utils


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=str,
        default="my_study",
        dest="study_name",
        help="Study name.",
    )
    return parser.parse_args()


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


def _main():
    args = _parse_args()

    output_manager = io_utils.OutputManager(None, base_dir="outputs")
    db_path = output_manager.file_path("optuna.db")
    db_url = f"sqlite:///{db_path}"
    storage = optuna.storages.RDBStorage(db_url)

    _show_studies(storage)

    study = optuna.load_study(study_name=f"{args.study_name}", storage=storage)

    _show_first_trials(study)
    _show_best_trial(study)


if __name__ == "__main__":
    _main()
