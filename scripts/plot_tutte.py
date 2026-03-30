from diff_tissue.app import config, io_utils, parameters, tutte


def _main():
    params = parameters.get_params_from_cli()

    output = io_utils.OutputManager(
        tutte.OUTPUT_TYPE_DIR, base_dir=config.load_cfg()["outputs_base_dir"]
    )

    tutte.plot(params, output)


if __name__ == "__main__":
    _main()
