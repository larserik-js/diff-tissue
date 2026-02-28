import argparse
from dataclasses import fields
from functools import cached_property
from typing import Any

from ..core.jax_bootstrap import struct


@struct.dataclass
class Params:
    system: str = struct.field(
        default="voronoi",
        pytree_node=False,
        metadata={
            "help": "Initial polygon configuration.",
            "choices": ["voronoi", "full", "single"],
            "cli_flag": "system",
        },
    )
    shape: str = struct.field(
        default="petal",
        pytree_node=False,
        metadata={
            "help": "Shape of target boundary.",
            "choices": ["petal", "trapezoid", "triangle", "nconv"],
            "cli_flag": "shape",
        },
    )
    knots: bool = struct.field(
        default=False,
        pytree_node=False,
        metadata={
            "help": (
                "Use knots as trainable parameters. If not set,"
                "train parameters for every individual polygon."
            ),
            "cli_flag": "knots",
        },
    )
    quiet: bool = struct.field(
        default=False,
        pytree_node=False,
        metadata={
            "help": "If set, no information on shape optimization is printed.",
            "cli_flag": "quiet",
        },
    )
    n_shape_steps: int = struct.field(
        default=500,
        pytree_node=True,
        metadata={"help": "Number of shape steps.", "cli_flag": "ssteps"},
    )
    n_growth_steps: int = struct.field(
        default=500,
        pytree_node=True,
        metadata={"help": "Number of growth steps.", "cli_flag": "gsteps"},
    )
    areas_loss_weight: float = struct.field(
        default=100.0,
        pytree_node=True,
        metadata={"help": "Areas loss weight.", "cli_flag": "arlw"},
    )
    angles_loss_weight: float = struct.field(
        default=200.0,
        pytree_node=True,
        metadata={"help": "Angles loss weight.", "cli_flag": "anlw"},
    )
    anisotropy_loss_weight: float = struct.field(
        default=300.0,
        pytree_node=True,
        metadata={"help": "Anisotropies loss weight.", "cli_flag": "elw"},
    )
    shape_loss_weight: float = struct.field(
        default=1.0,
        pytree_node=True,
        metadata={"help": "Shape loss weight.", "cli_flag": "slw"},
    )
    proximal_dist: float = struct.field(
        default=0.0,
        pytree_node=True,
        metadata={
            "help": "Distance from base for proximal polygons.",
            "cli_flag": "pd",
        },
    )
    max_area_scaling: float = struct.field(
        default=1.0,
        pytree_node=True,
        metadata={"help": "Maximum area scaling.", "cli_flag": "marsc"},
    )
    growth_scale: float = struct.field(
        default=5.0,
        pytree_node=True,
        metadata={"help": "Growth scale.", "cli_flag": "gsc"},
    )
    seed: int = struct.field(
        default=0,
        pytree_node=True,
        metadata={
            "help": "Random NumPy seed for reproducibility.",
            "cli_flag": "seed",
        },
    )

    def replace(self, **kwargs: Any) -> "Params":
        return self  # only for mypy, Flax runtime will override


def get_params_from_cli():
    parser = argparse.ArgumentParser()

    for field in fields(Params):
        kwargs = {
            "dest": field.name,
            "default": field.default,
            "help": field.metadata.get("help", ""),
        }

        if field.type is bool:
            if field.default is False:
                kwargs["action"] = "store_true"
            else:
                kwargs["action"] = "store_false"
        else:
            if field.type is not str:
                kwargs["type"] = field.type

            if "choices" in field.metadata:
                kwargs["choices"] = field.metadata["choices"]

        parser.add_argument(
            f"--{field.metadata['cli_flag']}",
            **kwargs,
        )

    args = vars(parser.parse_args())

    return Params(**args)


class _ParamStringFormatter:
    _formats = {
        "bool": "",
        "int": "d",
        "float": ".7f",
        "float64": ".7f",
        "str": "",
    }

    def __init__(self, params: Params):
        self._params = params

    @staticmethod
    def _get_val_type(val):
        type_ = type(val)
        type_str = type_.__name__
        return type_str

    def _format_param_val_str(self, name, val):
        val_type = self._get_val_type(val)
        format_ = self._formats[val_type]
        param_name_val = name + "=" + format(val, format_)
        if val_type == "float" or val_type == "float64":
            param_name_val = param_name_val.rstrip("0").rstrip(".")
        return param_name_val

    def _join_param_val_pairs(self):
        param_name_vals = []

        for field in fields(self._params):
            cli_flag = field.metadata.get("cli_flag")
            param_name_val = self._format_param_val_str(
                cli_flag, getattr(self._params, field.name)
            )
            param_name_vals.append(param_name_val)
        param_path_str = "__".join(param_name_vals)
        return param_path_str

    @cached_property
    def param_string(self):
        joined_param_names = self._join_param_val_pairs()
        return joined_param_names


def get_param_string(params):
    param_string = f"{_ParamStringFormatter(params).param_string}"
    return param_string
