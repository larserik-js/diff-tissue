from diff_tissue.app import parameters, tutte


def _main():
    params = parameters.get_params_from_cli()

    tutte.plot(params)


if __name__ == "__main__":
    _main()
