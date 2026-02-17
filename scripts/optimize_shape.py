from diff_tissue.app import parameters, shape_opt


def _main():
    params = parameters.get_params_from_cli()

    shape_opt.optimize_shape(params)
    

if __name__ == '__main__':
    _main()
