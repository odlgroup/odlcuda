// includes CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// thrust
#include <LCRUtils/cuda/disableThrustWarnings.h>
#include <thrust/transform.h>
#include <LCRUtils/cuda/enableThrustWarnings.h>

#include <iostream>

// ODL
#include <ODLpp/UFunc.h>
#include <ODLpp/DeviceVectorImpl.h>
#include <ODLpp/CudaVectorImpl.h>

// Utils
#include <math.h>

// clang-format off

// Instantiate the methods for each type
template <typename Tin, typename Tout, typename F>
void apply_transform(const CudaVectorImpl<Tin>& in, CudaVectorImpl<Tout>& out, F f) {
    thrust::transform(in._impl->begin(), in._impl->end(), out._impl->begin(), f);
}

#define X(fun, impl)                                                                                \
struct functor##fun{                                                                                \
    __device__ float operator()(float t) const {                                                    \
        return impl##(t);                                                                           \
    }                                                                                               \
};                                                                                                  \
void UFunc<float, float>::##fun##(const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out) {    \
    apply_transform(in, out, functor##fun{});                                                       \
}

ODLPP_FOR_EACH_FLOAT_UFUNC
#undef X