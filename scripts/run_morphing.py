from diff_tissue.app import config, experiments, parameters


def _main():
    params = parameters.get_params_from_cli()

    cfg = config.load_cfg("config.yml")
    paths = config.ProjectPaths(
        data_base_dir=cfg.data_base_dir,
        outputs_base_dir=cfg.outputs_base_dir,
    )
    experiments.run_morphing(params, paths)


if __name__ == "__main__":
    _main()
