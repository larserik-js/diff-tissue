import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diff_tissue import my_files, my_utils


def _find_max_val(arrays):
    max_val = -np.inf
    for array_ in arrays:
        max_val = np.max([max_val, array_.max()])
    return max_val


def _get_line(max_val):
    line = np.linspace(0.0, max_val, 1000)
    return line


def _add_eq_line(ax, line):
    ax.plot(line, line, color='k', ls='--')


def _add_zero_line(ax, line):
    ax.plot(line, np.zeros_like(line), color='k', ls='--')


def _add_lines(axs, col, max_val):
    line = _get_line(max_val)
    _add_eq_line(axs[0,col], line)
    _add_zero_line(axs[1,col], line)


def _plot(ax, final_goal_vals, final_vals, label):
    sorted_inds = np.argsort(final_goal_vals)
    ax.scatter(
        final_goal_vals[sorted_inds], final_vals[sorted_inds], s=4.0, alpha=0.5,
        label=label
    )


def _plot_residuals(ax, final_goal_vals, final_vals):
    residuals = final_goal_vals - final_vals
    ax.scatter(final_goal_vals, residuals, s=4.0)


def _plot_all(axs, df):
    all_arrays = [
        ['final_goal_area', 'final_area', 'best_goal_area'],
        ['final_goal_elongation', 'final_elongation',
         'best_goal_elongation']
    ]
    xlabels = ['Goal areas', 'Goal elongations']
    for i in range(2):
        arrays = [df[array_name].values for array_name in all_arrays[i]]
        max_val = _find_max_val(arrays)
        _add_lines(axs, i, max_val)

        _plot(axs[0,i], arrays[0], arrays[1], label='Final')
        _plot(axs[0,i], arrays[2], arrays[1], label='Best')

        _plot_residuals(axs[1,i], arrays[0], arrays[1])

        axs[0,i].set_xticklabels([])
        axs[1,i].set_xlabel(xlabels[i])
        axs[0,i].legend(loc='best')


def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()

    np.random.seed(params.numerical['seed'])

    input_file = my_files.get_output_params_file(params)
    df = pd.read_csv(input_file, sep='\t', index_col=0)

    fig, axs = plt.subplots(2,2)

    _plot_all(axs, df)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _main()
