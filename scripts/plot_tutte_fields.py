import yaml

from diff_tissue.app import experiments


def _load_cfg():
    with open("config.yml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def _main():
    experiments.plot_tutte_fields(
        outputs_base_dir=_load_cfg()["outputs_base_dir"]
    )


if __name__ == "__main__":
    _main()
