import pathlib

from diff_tissue import mapped_fields


def _main():
    shape = 'petal' # Only possibility as of now

    nx, ny = 100, 100

    output_dir = pathlib.Path('outputs')
    output_dir.mkdir(exist_ok=True)

    mapped_fields.run(shape, nx, ny, output_dir)


if __name__ == '__main__':
    _main()
