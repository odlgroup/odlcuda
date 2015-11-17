#pragma once

#include <stdint.h>
#include <sstream>
#include <type_traits>

#include <odlpp/cuda/DeviceVector.h>
#include <odlpp/cuda/CudaVectorImpl.h>

// clang-format off

//List of all Ufuncs
#define ODLPP_FOR_EACH_UFUNC \
    X(sin) \
    X(cos) \
    X(arcsin) \
    X(arccos) \
    X(log) \
    X(exp) \
    X(abs) \
    X(sign) \
    X(sqrt)

//Default to an error message
#define X(fun) template <typename Tin, typename Tout> void ufunc_##fun (const CudaVectorImpl<Tin>& in, CudaVectorImpl<Tout>& out) { throw std::domain_error("##fun UFunc not supported with this type"); }
ODLPP_FOR_EACH_UFUNC
#undef X

//Implementations for floats
#define X(fun) template<> void ufunc_##fun <float, float>(const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out);
ODLPP_FOR_EACH_UFUNC
#undef X
