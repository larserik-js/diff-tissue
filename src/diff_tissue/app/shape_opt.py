import pandas as pd

from ..core import my_utils
from . import io_utils, plotting


FINAL_TISSUES_DIR = 'final_tissues'
BEST_GROWTH_DIR = 'best_growth'


def save_output_params(tabular_output, params):
    df = pd.DataFrame(tabular_output)
    output_file = io_utils.get_output_params_file(params)
    df.to_csv(output_file, sep='\t', index=True, header=True)


def load_output_params(params):
    input_file = io_utils.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)

    best_goal_areas = my_utils.to_jax(df['best_goal_area'].values)
    best_goal_anisotropies = my_utils.to_jax(df['best_goal_anisotropy'].values)
    return best_goal_areas, best_goal_anisotropies


def plot_final_tissues(final_tissues, output, param_string, jax_arrays, params):
    figure = plotting.MorphFigure(jax_arrays, params)

    for t, vertices in enumerate(final_tissues):
        if t%10 == 0:
            fig_path = output.file_path(param_string, f'step={t:03d}.png')
            figure.save_plot(vertices, fig_path, enumerate=True)
    fig_path = output.file_path(param_string, f'step={t:03d}.png')
    figure.save_plot(vertices, fig_path, enumerate=True)


def plot_best_growth(
        growth_evolution, output, param_string, jax_arrays, params
    ):
    figure = plotting.MorphGrowthFigure(jax_arrays, params)

    for t, vertices in enumerate(growth_evolution):
        if t%10 == 0:
            fig_path = output.file_path(param_string, f'step={t:03d}.png')
            figure.save_plot(vertices, t, fig_path)
    fig_path = output.file_path(param_string, f'step={t:03d}.png')
    figure.save_plot(vertices, t, fig_path)
