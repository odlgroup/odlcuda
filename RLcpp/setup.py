# -*- coding: utf-8 -*-
"""

"""

from future import standard_library
standard_library.install_aliases()

from distutils.core import setup

# from RL import __version__

setup(name='RLcpp',
      # version=__version__,
      version='0.1.0',
      author='Jonas Adler',
      author_email='jonasadl@kth.se',
      url='https://gits-14.sys.kth.se/LCR/RLcpp',
      description='C++ backend for Regularization Library',
      license='GPLv3',
      packages=['RLcpp'],
      package_dir={'RLcpp': '.'},
      package_data={'RLcpp': ['*.*']})
