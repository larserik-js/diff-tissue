import argparse

from diff_tissue.app import config, param_search


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

    cfg = config.load_cfg("config.yml")
    paths = config.ProjectPaths(
        data_base_dir=cfg.data_base_dir,
        outputs_base_dir=cfg.outputs_base_dir,
    )
    param_search.inspect_param_search(paths, args.study_name)


if __name__ == "__main__":
    _main()
