import yaml

from diff_tissue.app import io_utils, parameters, tutte


def _load_cfg():
    with open("config.yml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def _main():
    params = parameters.get_params_from_cli()

    output = io_utils.OutputManager(
        tutte.OUTPUT_TYPE_DIR, base_dir=_load_cfg()["outputs_base_dir"]
    )

    tutte.plot(params, output)


if __name__ == "__main__":
    _main()
