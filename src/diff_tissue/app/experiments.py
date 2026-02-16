from ..core import my_utils
from . import io_utils, morphing, parameters


def run_morphing(params, base_dir='outputs'):
    jax_arrays = my_utils.get_jax_arrays(params)

    output = io_utils.OutputManager(morphing.OUTPUT_TYPE_DIR, base_dir=base_dir)

    param_string = f'{parameters.ParamStringFormatter(params).param_string}'
    cache_path = output.cache_path(f'{param_string}.pkl')

    growth_evolution = morphing.get_growth_evolution(
        cache_path, jax_arrays, params
    )

    morphing.save_figs(
        growth_evolution, output, param_string, jax_arrays, params
    )
