import argparse

from diff_tissue.app import config, searches


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


def _main():
    args = _parse_args()

    searches.inspect_param_search(
        args.study_name, outputs_base_dir=config.load_cfg()["outputs_base_dir"]
    )


if __name__ == "__main__":
    _main()
