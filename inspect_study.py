import argparse

import optuna


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n',
        type=str,
        dest='study_name',
        help='Study name.'
    )
    return parser.parse_args()


def _show_first_trials(study):
    df = study.trials_dataframe()
    print(df.head())
    print('')


def _show_best_trial(study):
    print(f'{'Best loss:':<30}{study.best_value}')
    print('')

    for param, val in study.best_params.items():
        print(f'{param + ':':<30}{val}')
    print('')


def _main():
    args = _parse_args()

    study = optuna.load_study(
        study_name=f'{args.study_name}',
        storage='sqlite:///optuna.db'
    )

    _show_first_trials(study)
    _show_best_trial(study)


if __name__ == '__main__':
    _main()
