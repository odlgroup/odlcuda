"""Exposes the odlcuda files to ODL as a plugin."""

from odlcuda import cu_ntuples


def tensor_space_impls():
    return {'odlcuda': cu_ntuples.CudaTensorSpace}


def tensor_space_impl_names():
    return tuple(tensor_space_impls().keys())
