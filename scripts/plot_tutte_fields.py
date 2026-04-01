from diff_tissue.app import config, experiments


def _main():
    cfg = config.load_cfg("config.yml")
    paths = config.ProjectPaths(cfg.data_base_dir, cfg.outputs_base_dir)
    experiments.plot_tutte_fields(paths)


if __name__ == "__main__":
    _main()
