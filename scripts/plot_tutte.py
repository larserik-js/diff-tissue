from diff_tissue.app import config, parameters, tutte


def _main():
    params = parameters.get_params_from_cli()

    cfg = config.load_cfg("config.yml")
    paths = config.ProjectPaths(
        data_base_dir=cfg.data_base_dir,
        outputs_base_dir=cfg.outputs_base_dir,
    )
    output_dir = paths.make_subdir(
        paths.outputs_base_dir, tutte.OUTPUT_TYPE_DIR
    )

    tutte.plot(params, output_dir)


if __name__ == "__main__":
    _main()
