import argparse


class Params:
    def __init__(self):
        self._args = self._parse_args()
        self.system = self._args.system
        self.shape = self._args.shape
        self.knots = self._args.knots
        self.quiet = self._args.quiet
        self.numerical = {
            'n_shape_steps': self._args.ssteps,
            'n_growth_steps': self._args.gsteps,
            'areas_loss_weight': self._args.arlw,
            'angles_loss_weight': self._args.anlw,
            'elongation_loss_weight': self._args.elw,
            'max_area_scaling': self._args.marsc,
            'proximal_dist': self._args.pd,
            'growth_scale': self._args.gsc,
            'seed': self._args.seed
        }
        self.all = vars(self._args)

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
