import pickle

from diff_tissue import io_utils, mapped_fields


def _main():
    shape = 'petal' # Only possibility as of now

    nx, ny = 100, 100

    meshes_file = io_utils.get_output_path(f'meshes_{shape}.pkl')

    mapped_fields_ = mapped_fields.run(shape, nx, ny, meshes_file)

    mapped_fields_file = io_utils.get_output_path(f'mapped_fields_{shape}.pkl')
    with open(mapped_fields_file, 'wb') as f:
        pickle.dump(mapped_fields_, f)


if __name__ == '__main__':
    _main()
