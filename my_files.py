import os
from pathlib import Path
import pickle


class _Output:
    _formats = {'bool': '',
                'int': 'd',
                'float': '.7f',
                'float64': '.7f',
                'str': ''}

    def __init__(self, output_type_dir, params):
        self._project_dir = self._get_project_dir()
        self._output_dir = self._project_dir / 'output'
        self._output_type_dir = self._output_dir / output_type_dir
        self._params = params
        self._param_path = self._make_param_path()

    def _get_project_dir(self):
        project_dir = os.path.abspath(os.path.dirname(__file__))
        return Path(project_dir)

    @staticmethod
    def _get_val_type(val):
        type_ = type(val)
        type_str = type_.__name__
        return type_str

    def _format_param_val_str(self, name, val):
        val_type = self._get_val_type(val)
        format_ = self._formats[val_type]
        param_name_val = name + '=' + format(val, format_)
        if val_type == 'float' or val_type == 'float64':
            param_name_val = param_name_val.rstrip('0').rstrip('.')
        return param_name_val

    def _concatenate_param_val_pairs(self):
        param_name_vals = []
        for name, val in self._params.all.items():
            param_name_val = self._format_param_val_str(name, val)
            param_name_vals.append(param_name_val)

        param_path_str = '_'.join(param_name_vals)
        return param_path_str

    def _make_param_path(self):
        param_path_str = self._concatenate_param_val_pairs()
        param_path = self._output_type_dir / param_path_str
        return param_path

    def get_output_type_dir(self):
        return self._output_type_dir


class OutputDir(_Output):
    def __init__(self, output_type_dir, params):
        super().__init__(output_type_dir, params)
        self._make()

    def _make(self):
        self._param_path.mkdir(exist_ok=True)

    def get_path(self):
        return self._param_path


class OutputFile(_Output):
    def __init__(self, output_type_dir, suffix, params):
        super().__init__(output_type_dir, params)
        self._path = self._param_path.with_name(self._param_path.name + suffix)

    def get_path(self):
        return self._path


class DataHandler:
    def __init__(self, file):
        self._file_path = file.get_path()

    def _load_pkl(self):
        with open(self._file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _save_pkl(self, data):
        with open(self._file_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        if self._file_path.suffix == '.pkl':
            data = self._load_pkl()
        else:
            raise NotImplementedError
        return data

    def save(self, data):
        if self._file_path.suffix == '.pkl':
            self._save_pkl(data)
        else:
            raise NotImplementedError


def get_output_params_file(params):
    return OutputFile('output_params', '.txt', params).get_path()
