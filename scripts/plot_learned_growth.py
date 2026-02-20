from diff_tissue.app import experiments, parameters


def _main():
    params = parameters.get_params_from_cli()

    experiments.run_learned_growth(params)


if __name__ == "__main__":
    _main()
