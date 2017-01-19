"""Setup script for odlcuda."""

from future import standard_library
standard_library.install_aliases()

from setuptools import setup


setup(name='odlcuda',
      version='0.5.0',
      author='Jonas Adler',
      author_email='jonasadl@kth.se',
      url='https://github.com/odlgroup/odlcuda',
      description='C++ backend for odl',
      license='GPLv3',
      install_requires=['odl >= 0.5.3'],
      packages=['odlcuda'],
      package_dir={'odlcuda': '.'},
      package_data={'' : ['*.pyd', '*.so']},
      entry_points={'odl.space': ['odlcuda = odlcuda.odl_plugin']})
