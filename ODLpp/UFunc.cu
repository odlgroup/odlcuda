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

// Instantiate the methods for each type
template <typename Tin, typename Tout, typename F>
void apply_transform(const CudaVectorImpl<Tin>& in, CudaVectorImpl<Tout>& out, F f) {
    thrust::transform(in._impl->begin(), in._impl->end(), out._impl->begin(), f);
}

struct functor_sign {
    __device__ float operator()(float t) const {
        return static_cast<float>((t > 0.0f) - (t < 0.0f));
    }
};
template <>
void ufunc_sign(const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out) {
    apply_transform(in, out, functor_sign{});
}

// clang-format off
#define ODL_DEFINE_FLOAT_UFUNC(fun, impl)                                                       \
struct functor_##fun{                                                                           \
    __device__ float operator()(float t) const {                                                \
        return impl(t);                                                                         \
    }                                                                                           \
};                                                                                              \
template <> void ufunc_##fun (const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out) {    \
    apply_transform(in, out, functor_##fun{});                                                  \
}

ODL_DEFINE_FLOAT_UFUNC(sin, sinf)
ODL_DEFINE_FLOAT_UFUNC(cos, cosf)
ODL_DEFINE_FLOAT_UFUNC(arcsin, asinf)
ODL_DEFINE_FLOAT_UFUNC(arccos, acosf)
ODL_DEFINE_FLOAT_UFUNC(log, logf)
ODL_DEFINE_FLOAT_UFUNC(exp, expf)
ODL_DEFINE_FLOAT_UFUNC(abs, fabsf)
ODL_DEFINE_FLOAT_UFUNC(sqrt, sqrtf)

#undef ODL_DEFINE_FLOAT_UFUNC