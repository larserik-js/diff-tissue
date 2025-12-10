from abc import ABC, abstractmethod
from functools import cached_property
import os
from pathlib import Path
import pickle


class _Output(ABC):
    _formats = {'bool': '',
                'int': 'd',
                'float': '.7f',
                'float64': '.7f',
                'str': ''}

    def __init__(self, output_type_dir_name, params):
        self._output_type_dir_name = output_type_dir_name
        self._params = params
        self._set_param_names()

    @abstractmethod
    def _set_param_names(self):
        pass

    @cached_property
    def _project_dir(self):
        project_dir = os.path.abspath(os.path.dirname(__file__))
        return Path(project_dir)

    @cached_property
    def _output_type_dir(self):
        output_type_dir = (
            self._project_dir / 'output' / self._output_type_dir_name
        )
        return output_type_dir

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
        for name in self._param_names:
            param_name_val = self._format_param_val_str(
                name, self._params.all[name]
            )
            param_name_vals.append(param_name_val)

        param_path_str = '_'.join(param_name_vals)
        return param_path_str

    def _make_param_path(self):
        param_path_str = self._concatenate_param_val_pairs()
        param_path = self._output_type_dir / param_path_str
        return param_path


class OutputDir(_Output):
    def __init__(self, output_type_dir, params):
        super().__init__(output_type_dir, params)
        self._make()

    def _set_param_names(self):
        self._param_names = self._params.all.keys()

    @cached_property
    def path(self):
        path = self._make_param_path()
        return path

    def _make(self):
        self.path.mkdir(exist_ok=True)


class OutputFile(_Output):
    def __init__(self, output_type_dir, suffix, params):
        super().__init__(output_type_dir, params)
        self._suffix = suffix

    def _set_param_names(self):
        self._param_names = self._params.all.keys()

    @cached_property
    def path(self):
        params_path = self._make_param_path()
        path = params_path.with_name(params_path.name + self._suffix)
        return path


class ArraysFile(_Output):
    def __init__(self, output_type_dir, suffix, params):
        super().__init__(output_type_dir, params)
        self._suffix = suffix

    def _set_param_names(self):
        self._param_names = ['system', 'shape', 'seed']

    @cached_property
    def path(self):
        param_path = self._make_param_path()
        path = param_path.with_name(param_path.name + self._suffix)
        return path


class DataHandler:
    def __init__(self, file):
        self._file_path = file.path

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
    return OutputFile('output_params', '.txt', params).path
