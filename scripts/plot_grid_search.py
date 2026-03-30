import argparse

from diff_tissue.app import config, grid_search


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


def _main():
    args = _parse_args()

    paths = config.ProjectPaths(
        data_base_dir=config.load_cfg("config.yml").data_base_dir,
        outputs_base_dir=config.load_cfg("config.yml").outputs_base_dir,
    )
    grid_search.plot(args.study_name, paths)


if __name__ == "__main__":
    _main()
