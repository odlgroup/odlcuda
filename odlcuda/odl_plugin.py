"""Exposes the odlcuda files to ODL as a plugin."""

from odlcuda import cu_ntuples

def ntuples_impls():
    return {'cuda': cu_ntuples.CudaNtuples}

def fn_impls():
    return {'cuda': cu_ntuples.CudaFn}
