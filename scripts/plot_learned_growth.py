from diff_tissue.app import io_utils, learned_growth, parameters


def _main():
    params = parameters.get_params_from_cli()

    results = learned_growth.run(params)

    output = io_utils.OutputManager(
        learned_growth.OUTPUT_TYPE_DIR, base_dir="outputs"
    )

    param_string = parameters.get_param_string(params)
    learned_growth.plot(results, output, param_string)


if __name__ == "__main__":
    _main()
