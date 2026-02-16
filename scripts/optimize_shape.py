import pandas as pd

from diff_tissue.core import my_utils, shape_opt
from diff_tissue.app import io_utils, parameters


def _save_final_tissues(final_tissues, params):
    output_path = io_utils.OutputFile('final_tissues', '.pkl', params).path
    io_utils.save_pkl(output_path, final_tissues)


def _save_output_params(param_dict, params):
    df = pd.DataFrame(param_dict)
    output_file = io_utils.get_output_params_file(params)
    df.to_csv(output_file, sep='\t', index=True, header=True)


@my_utils.timer
def _main():
    params = parameters.get_params_from_cli()

    _, final_tissues, _, tabular_output = shape_opt.run(params)
    
    _save_final_tissues(final_tissues, params)
    _save_output_params(tabular_output, params)


if __name__ == '__main__':
    _main()
