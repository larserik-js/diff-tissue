import yaml

from diff_tissue.app import experiments, parameters


def _load_cfg():
    with open("config.yml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def _main():
    params = parameters.get_params_from_cli()

    experiments.run_shape_opt(
        params, outputs_base_dir=_load_cfg()["outputs_base_dir"]
    )


if __name__ == "__main__":
    _main()
