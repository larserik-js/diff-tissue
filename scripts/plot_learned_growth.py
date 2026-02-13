from diff_tissue import io_utils, learned_growth, my_utils, parameters


def _main():
    params = parameters.get_params_from_cli()

    jax_arrays = my_utils.get_jax_arrays(params)

    results = learned_growth.run(jax_arrays, params)

    output_dir = io_utils.OutputDir('learned_growth', params).path

    learned_growth.plot(results, output_dir)


if __name__ == '__main__':
    _main()
