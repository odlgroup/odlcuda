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

"""UFuncs for ODL vectors.

These functions are internal and should only be used as methods on
`NtuplesBaseVector` type spaces.

See `numpy.ufuncs
<http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
for more information.

Notes
-----
The default implementation of these methods make heavy use of the
``NtuplesBaseVector.__array__`` to extract a `numpy.ndarray` from the vector,
and then apply a ufunc to it. Afterwards, ``NtuplesBaseVector.__array_wrap__``
is used to re-wrap the data into the appropriate space.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


import odl


__all__ = ('CudaNtuplesUFuncs',)


# Optimizations for CUDA
def _make_nullary_fun(name):
    def fun(self):
        return getattr(self.vector.data, name)()

    fun.__doc__ = getattr(odl.util.ufuncs.NtuplesBaseUfuncs, name).__doc__
    fun.__name__ = name
    return fun


def _make_unary_fun(name):
    def fun(self, out=None):
        if out is None:
            out = self.vector.space.element()
        getattr(self.vector.data, name)(out.data)
        return out

    fun.__doc__ = getattr(odl.util.ufuncs.NtuplesBaseUfuncs, name).__doc__
    fun.__name__ = name
    return fun


class CudaNtuplesUFuncs(odl.util.ufuncs.NtuplesBaseUfuncs):

    """UFuncs for `CudaNtuplesVector` objects.

    Internal object, should not be created except in `CudaNtuplesVector`.
    """

    # Ufuncs
    sin = _make_unary_fun('sin')
    cos = _make_unary_fun('cos')
    arcsin = _make_unary_fun('arcsin')
    arccos = _make_unary_fun('arccos')
    log = _make_unary_fun('log')
    exp = _make_unary_fun('exp')
    absolute = _make_unary_fun('absolute')
    sign = _make_unary_fun('sign')
    sqrt = _make_unary_fun('sqrt')

    # Reductions
    sum = _make_nullary_fun('sum')
    prod = _make_nullary_fun('prod')
    min = _make_nullary_fun('min')
    max = _make_nullary_fun('max')
