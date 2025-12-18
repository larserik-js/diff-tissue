import optuna

import my_utils, shape_opt


def objective_f(trial):
    params = my_utils.Params()

    params.numerical['growth_steps'] = trial.suggest_int(
        'growth steps', 50, 2000
    )
    params.numerical['area_loss_weight'] = trial.suggest_float(
        'area loss weight', 1.0, 1e4, log=True
    )
    params.numerical['angle_loss_weight'] = trial.suggest_float(
        'angle loss weight', 1.0, 1e4, log=True
    )
    params.numerical['aspect_ratio_loss_weight'] = trial.suggest_float(
        'aspect ratio loss weight', 1.0, 1e4, log=True
    )
    params.numerical['max_area_scaling'] = trial.suggest_float(
        'max area scaling', 2.0, 9.0
    )
    params.numerical['seed'] = 3
    params.quiet = True

    loss, _, _ = shape_opt.run(params)

    return loss


def _main():
    study = optuna.create_study(
        study_name='my_study',
        direction='minimize',
        storage='sqlite:///optuna.db',
        load_if_exists=True,
    )

    study.optimize(objective_f, n_trials=1000, n_jobs=8)

    print('Best loss:', study.best_value)
    print('Best params:', study.best_params)


if __name__ == '__main__':
    _main()
