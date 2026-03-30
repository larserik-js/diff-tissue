from diff_tissue.app import config, experiments


def _main():
    experiments.plot_tutte_fields(
        outputs_base_dir=config.load_cfg()["outputs_base_dir"]
    )


if __name__ == "__main__":
    _main()
