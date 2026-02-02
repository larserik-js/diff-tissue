import argparse

from flax import struct


@struct.dataclass
class Params:
    system: str = struct.field(pytree_node=False)
    shape: str = struct.field(pytree_node=False)
    knots: bool = struct.field(pytree_node=False)
    quiet: bool = struct.field(pytree_node=False)
    n_shape_steps: int
    n_growth_steps: int
    areas_loss_weight: float
    angles_loss_weight: float
    elongation_loss_weight: float
    max_area_scaling: float
    proximal_dist: float
    growth_scale: float
    seed: int

    def get_names(self):
        names = list(self.__dataclass_fields__.keys())
        return names


class _Params:
    def __init__(self):
        self._args = self._parse_args()
        self._short_to_long = {
            'system': 'system',
            'shape': 'shape',
            'knots': 'knots',
            'quiet': 'quiet',
            'ssteps': 'n_shape_steps',
            'gsteps': 'n_growth_steps',
            'arlw': 'areas_loss_weight',
            'anlw': 'angles_loss_weight',
            'elw': 'elongation_loss_weight',
            'marsc': 'max_area_scaling',
            'pd': 'proximal_dist',
            'gsc': 'growth_scale',
            'seed': 'seed'
        }
        dict_items = vars(self._args)
        self._dataclass_kwargs = {
            self._short_to_long[short]: val for short, val in dict_items.items()
        }
        self.params_from_datacls = Params(**self._dataclass_kwargs)

    def _parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--system',
            type=str,
            choices=['voronoi', 'full', 'single'],
            default='voronoi',
            help='Initial polygon configuration.'
        )

        parser.add_argument(
            '--shape',
            type=str,
            choices=['petal', 'trapezoid', 'triangle', 'nconv'],
            default='petal',
            help='Type of outer shape.'
        )

        parser.add_argument(
            '--knots',
            action='store_true',
            help=(
                'Use knots as trainable parameters.' +
                'If not set, train parameters for every individual polygon.'
            )
        )

        parser.add_argument(
            '--quiet',
            action='store_true',
            help=('If set, no information on shape optimization is printed.')
        )

        parser.add_argument(
            '--ssteps',
            type=int,
            default=200,
            help='Number of shape steps.'
        )

        parser.add_argument(
            '--gsteps',
            type=int,
            default=500,
            help='Number of growth steps.'
        )

        parser.add_argument(
            '--arlw',
            type=float,
            default=100.0,
            help='Areas loss weight.'
        )

        parser.add_argument(
            '--anlw',
            type=float,
            default=200.0,
            help='Angles loss weight.'
        )

        parser.add_argument(
            '--elw',
            type=float,
            default=300.0,
            help='Elongations loss weight.'
        )

        parser.add_argument(
            '--marsc',
            type=float,
            default=9.0,
            help='Maximal area scaling.'
        )

        parser.add_argument(
            '--pd',
            type=float,
            default=0.0,
            help='Distance from base for proximal polygons.'
        )

        parser.add_argument(
            '--gsc',
            type=float,
            default=5.0,
            help='Growth scale.'
        )

        parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='Random NumPy seed for reproducibility.'
        )
        args = parser.parse_args()
        return args


def get_params_from_cli():
    return _Params().params_from_datacls
