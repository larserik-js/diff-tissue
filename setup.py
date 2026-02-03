from setuptools import find_packages, setup


setup(
    name='diff_tissue',
    package_dir={'': 'src'},
    packages=find_packages(where='src')
)
