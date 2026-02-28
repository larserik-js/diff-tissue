from diff_tissue.app import io_utils, parameters, shape_opt


def _main():
    params = parameters.get_params_from_cli()

    output = io_utils.OutputManager(
        shape_opt.OUTPUT_TYPE_DIR, base_dir="outputs"
    )

    _ = shape_opt.get_sim_states(params, output)


if __name__ == "__main__":
    _main()
