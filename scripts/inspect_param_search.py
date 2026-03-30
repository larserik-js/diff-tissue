import argparse

import yaml

from diff_tissue.app import searches


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=str,
        default="my_study",
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

    searches.inspect_param_search(
        args.study_name, outputs_base_dir=_load_cfg()["outputs_base_dir"]
    )


if __name__ == "__main__":
    _main()
