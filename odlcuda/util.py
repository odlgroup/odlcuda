# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""CUDA space utils."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


__all__ = ('as_numba_arr',)


def as_numba_arr(el):
    """Convert ``el`` to numba array."""
    import numba.cuda
    import ctypes

    gpu_data = numba.cuda.cudadrv.driver.MemoryPointer(
        context=numba.cuda.current_context(),
        pointer=ctypes.c_ulong(el.data_ptr),
        size=el.size)

    return numba.cuda.cudadrv.devicearray.DeviceNDArray(
        shape=el.shape,
        strides=(el.dtype.itemsize,),
        dtype=el.dtype,
        gpu_data=gpu_data)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
