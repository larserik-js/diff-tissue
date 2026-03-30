from . import io_utils


def load_cfg():
    return io_utils.load_yaml("config.yml")
