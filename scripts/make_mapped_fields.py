from diff_tissue import mapped_fields


def _main():
    shape = 'petal' # Only possibility as of now

    nx, ny = 100, 100

    mapped_fields.run(shape, nx, ny)


if __name__ == '__main__':
    _main()
