import argparse

from diff_tissue.app import searches


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=str,
        default="grid_search",
        dest="study_name",
        help="Study name.",
    )
    return parser.parse_args()


def _main():
    args = _parse_args()

    searches.inspect_grid_search(args.study_name)


if __name__ == "__main__":
    _main()
