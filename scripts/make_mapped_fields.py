from diff_tissue import mapped_fields, my_files


def _main():
    shape = 'petal' # Only possibility as of now

    nx, ny = 100, 100

    mapped_fields.run(shape, nx, ny, my_files.BASE_OUTPUT_DIR)


if __name__ == '__main__':
    _main()
