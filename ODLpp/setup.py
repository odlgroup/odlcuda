# -*- coding: utf-8 -*-
"""

"""

from future import standard_library
standard_library.install_aliases()

from distutils.core import setup

# from odl import __version__

setup(name='odlpp',
      # version=__version__,
      version='0.1.0',
      author='Jonas Adler',
      author_email='jonasadl@kth.se',
      url='https://gits-14.sys.kth.se/LCR/ODLpp',
      description='C++ backend for ODL',
      license='GPLv3',
      packages=['odlpp'],
      package_dir={'odlpp': '.'},
      package_data={'odlpp': ['*.*']})
