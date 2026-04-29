from diff_tissue.app import param_search as param_search_app
from diff_tissue.app import config


def _main():
    cfg = config.load_cfg("config.yml")
    paths = config.ProjectPaths(
        data_base_dir=cfg.data_base_dir,
        outputs_base_dir=cfg.outputs_base_dir,
    )

    param_search_app.run(paths)


if __name__ == "__main__":
    _main()
