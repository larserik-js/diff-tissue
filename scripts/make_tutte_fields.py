import pickle

from diff_tissue import io_utils, tutte_fields


def _main():
    shape = 'petal' # Only possibility as of now

    nx, ny = 100, 100

    meshes_file = io_utils.get_output_path(f'meshes_{shape}.pkl')

    tutte_fields_ = tutte_fields.run(shape, nx, ny, meshes_file)

    tutte_fields_file = io_utils.get_output_path(f'tutte_fields_{shape}.pkl')
    with open(tutte_fields_file, 'wb') as f:
        pickle.dump(tutte_fields_, f)


if __name__ == '__main__':
    _main()
