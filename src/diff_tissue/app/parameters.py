import argparse
from dataclasses import fields
from functools import cached_property
from typing import Any

from ..core.jax_bootstrap import struct


@struct.dataclass
class Params:
    system: str = struct.field(
        default="few",
        pytree_node=False,
        metadata={
            "help": "Initial polygon configuration.",
            "choices": ["few", "many", "full", "single"],
            "cli_flag": "system",
        },
    )
    shape: str = struct.field(
        default="trapezoid",
        pytree_node=False,
        metadata={
            "help": "Shape of target boundary.",
            "choices": [
                "trapezoid",
                "petal",
                "long_petal",
                "nconv",
                "complex_nconv",
            ],
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
    trapezoid_angle: float = struct.field(
        default=75.0,
        pytree_node=True,
        metadata={
            "help": (
                "Angle between the base and the right leg of the "
                "isosceles trapezoid."
            ),
            "cli_flag": "tran",
        },
    )
    n_morph_steps: int = struct.field(
        default=500,
        pytree_node=True,
        metadata={"help": "Number of morph steps.", "cli_flag": "msteps"},
    )
    areas_pot_weight: float = struct.field(
        default=5.0,
        pytree_node=True,
        metadata={"help": "Areas potential weight.", "cli_flag": "arpw"},
    )
    angles_pot_weight: float = struct.field(
        default=13.0,
        pytree_node=True,
        metadata={"help": "Angles potential weight.", "cli_flag": "anpw"},
    )
    anisotropies_pot_weight: float = struct.field(
        default=50.0,
        pytree_node=True,
        metadata={
            "help": "Anisotropies potential weight.",
            "cli_flag": "aspw",
        },
    )
    init_lr: float = struct.field(
        default=0.01,
        pytree_node=True,
        metadata={
            "help": "Initial learning rate for shape optimization.",
            "cli_flag": "init_lr",
        },
    )
    shape_loss_weight: float = struct.field(
        default=1.0,
        pytree_node=True,
        metadata={"help": "Shape loss weight.", "cli_flag": "slw"},
    )
    poly_id_configuration: int = struct.field(
        default=0,
        pytree_node=False,
        metadata={
            "help": ("Polygon identity configuration."),
            "choices": [0, 1],
            "cli_flag": "id",
        },
    )
    poly_id_loss_weight: float = struct.field(
        default=0.4,
        pytree_node=True,
        metadata={"help": "Poly identity loss weight.", "cli_flag": "ilw"},
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


def format_float_to_str(float_):
    rounded_float = round(float_, 8)
    float_str = str(rounded_float)
    if float_str[0] == "-":
        float_str = f"m{float_str[1:]}"
    return float_str.replace(".", "p")


class _ParamStringFormatter:
    def __init__(self, params: Params):
        self._params = params

    def _format_param_val_str(self, name, val):
        val_type = type(val).__name__
        if val_type == "float" or val_type == "float64":
            val_str = format_float_to_str(val)
        else:
            val_str = str(val)
        param_name_val = f"{name}={val_str}"
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
