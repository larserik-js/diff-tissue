import optuna

from diff_tissue import io_utils, parameters, shape_opt


def objective_f(trial):
    params = parameters.Params()

    params = params.replace(
        n_growth_steps = trial.suggest_int('growth steps', 50, 2000)
    )
    params = params.replace(
        areas_loss_weight = trial.suggest_float(
            'area loss weight', 1.0, 1e4, log=True
        )
    )
    params = params.replace(
        angles_loss_weight = trial.suggest_float(
            'angle loss weight', 1.0, 1e4, log=True
        )
    )
    params = params.replace(
        elongation_loss_weight = trial.suggest_float(
            'elongation loss weight', 1.0, 1e4, log=True
        )
    )
    params = params.replace(
        max_area_scaling = trial.suggest_float('max area scaling', 0.5, 2.0)
    )
    params = params.replace(seed = 3)
    params = params.replace(quiet = True)

    loss, _, _, _ = shape_opt.run(params)

    return loss


def _main():
    output_manager = io_utils.OutputManager(None)
    db_path = output_manager.file_path('optuna.db')

    db_url = f'sqlite:///{db_path}'
    study = optuna.create_study(
        study_name='my_study',
        direction='minimize',
        storage=db_url,
        load_if_exists=True,
    )

    study.optimize(objective_f, n_trials=1000, n_jobs=8)

    print('Best loss:', study.best_value)
    print('Best params:', study.best_params)


if __name__ == '__main__':
    _main()
