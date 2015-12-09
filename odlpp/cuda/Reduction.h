#pragma once

#include <stdint.h>
#include <sstream>
#include <type_traits>

#include <odlpp/cuda/DeviceVector.h>
#include <odlpp/cuda/CudaVectorImpl.h>

// clang-format off

//List of all Reductions
#define ODL_CUDA_FOR_EACH_REDUCTION \
    X(sum) \
    X(prod) \
    X(min) \
    X(max)

//Default to an error message
#define X(fun) template <typename T> T reduction_##fun (const CudaVectorImpl<T>& v) { throw std::domain_error("##fun reduction not supported with this type"); }
ODL_CUDA_FOR_EACH_REDUCTION
#undef X

//Implementations for floats
#define X(fun) template<> float reduction_##fun <float>(const CudaVectorImpl<float>& v);
ODL_CUDA_FOR_EACH_REDUCTION
#undef X
