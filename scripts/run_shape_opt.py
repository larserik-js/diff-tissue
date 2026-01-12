import jax
import pandas as pd

from diff_tissue import my_files, my_utils, shape_opt


def _save_final_tissues(final_tissues, params):
    output_file = my_files.OutputFile('final_tissues', '.pkl', params)
    data_handler = my_files.DataHandler(output_file)
    data_handler.save(final_tissues)


def _save_output_params(param_dict, params):
    df = pd.DataFrame(param_dict)
    output_file = my_files.get_output_params_file(params)
    df.to_csv(output_file, sep='\t', index=True, header=True)


@my_utils.timer
def _main():
    jax.config.update('jax_enable_x64', True)

    params = my_utils.Params()
    _, final_tissues, tabular_output = shape_opt.run(params)
    
    _save_final_tissues(final_tissues, params)
    _save_output_params(tabular_output, params)


if __name__ == '__main__':
    _main()
