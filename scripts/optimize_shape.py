from diff_tissue.app import config, parameters, shape_opt


def _main():
    params = parameters.get_params_from_cli()

    output = config.OutputManager(
        shape_opt.OUTPUT_TYPE_DIR,
        base_dir=config.load_cfg("config.yml").outputs_base_dir,
    )

    _ = shape_opt.get_sim_states(params, output)


if __name__ == "__main__":
    _main()
