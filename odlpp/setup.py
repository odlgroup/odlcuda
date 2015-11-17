# -*- coding: utf-8 -*-
"""

"""

from future import standard_library
standard_library.install_aliases()

from distutils.core import setup


setup(name='odlpp',
      version='0.2.0',
      author='Jonas Adler',
      author_email='jonasadl@kth.se',
      url='https://github.com/odlgroup/odlpp',
      description='C++ backend for odl',
      license='GPLv3',
      packages=['odlpp'],
      package_dir={'odlpp': '.'},
      package_data={'odlpp': ['*.*']})
