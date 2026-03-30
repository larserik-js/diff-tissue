import argparse

import yaml

from diff_tissue.app import grid_search


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=str,
        default="base_model",
        dest="study_name",
        help="Study name.",
    )
    return parser.parse_args()


def _load_cfg():
    with open("config.yml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def _main():
    args = _parse_args()

    grid_search.plot(args.study_name, _load_cfg()["outputs_base_dir"])


if __name__ == "__main__":
    _main()
