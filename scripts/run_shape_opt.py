from diff_tissue.app import config, experiments, parameters


def _main():
    params = parameters.get_params_from_cli()

    experiments.run_shape_opt(
        params, outputs_base_dir=config.load_cfg()["outputs_base_dir"]
    )


if __name__ == "__main__":
    _main()
