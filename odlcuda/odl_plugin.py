"""Exposes the odlcuda files to ODL as a plugin."""

from odlcuda import cu_ntuples


def fn_impls():
    return {'cuda': cu_ntuples.CudaFn}


def fn_impl_names():
    return tuple(fn_impls().keys())
