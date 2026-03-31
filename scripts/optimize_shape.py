from diff_tissue.app import config, parameters, shape_opt


def _main():
    params = parameters.get_params_from_cli()

    cfg = config.load_cfg("config.yml")
    paths = config.ProjectPaths(cfg.data_base_dir, cfg.outputs_base_dir)

    _ = shape_opt.get_sim_states(params, paths)


if __name__ == "__main__":
    _main()
