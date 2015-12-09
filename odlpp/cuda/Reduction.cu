// includes CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// thrust
#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <iostream>

// ODL
#include <odlpp/cuda/Reduction.h>
#include <odlpp/cuda/DeviceVectorImpl.h>
#include <odlpp/cuda/CudaVectorImpl.h>

// Utils
#include <math.h>

// Instantiate the methods for each type
template <>
float reduction_sum(const CudaVectorImpl<float>& v) {
    return thrust::reduce(v._impl->begin(), v._impl->end());
}

template <>
float reduction_prod(const CudaVectorImpl<float>& v) {
    return thrust::reduce(v._impl->begin(), v._impl->end(), 1, thrust::multiplies<float>());
}

template <>
float reduction_min(const CudaVectorImpl<float>& v) {
    return *thrust::min_element(v._impl->begin(), v._impl->end());
}

template <>
float reduction_max(const CudaVectorImpl<float>& v) {
    return *thrust::max_element(v._impl->begin(), v._impl->end());
}