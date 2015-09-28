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

#define ODLPP_FOR_EACH_FLOAT_UFUNC \
    X(sin, sinf) \
    X(cos, cosf) \
    X(arcsin, asinf) \
    X(arccos, acosf) \
    X(log, logf) \
    X(exp, expf) \
    X(abs, fabsf) \
    X(sqrt, sqrtf)

// Instantiate the methods for each type
template <typename Tin, typename Tout, typename F>
void apply_transform(const CudaVectorImpl<Tin>& in, CudaVectorImpl<Tout>& out, F f) {
    thrust::transform(in._impl->begin(), in._impl->end(), out._impl->begin(), f);
}

struct functorsign{
    __device__ float operator()(float t) const {
        return static_cast<float>((t > 0.0f) - (t < 0.0f));
    }
};
void ufuncsign(const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out) {
    apply_transform(in, out, functorsign{});
}

#define X(fun, impl)                                                                                \
struct functor##fun{                                                                                \
    __device__ float operator()(float t) const {                                                    \
        return impl##(t);                                                                           \
    }                                                                                               \
};                                                                                                  \
void ufunc##fun##(const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out) {                    \
    apply_transform(in, out, functor##fun{});                                                       \
}

ODLPP_FOR_EACH_FLOAT_UFUNC
#undef X