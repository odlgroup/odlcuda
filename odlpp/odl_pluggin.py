from . import cu_ntuples

def ntuples_impls():
    return {'cuda': cu_ntuples.CudaNtuples}

def fn_impls():
    return {'cuda': cu_ntuples.CudaFn}
