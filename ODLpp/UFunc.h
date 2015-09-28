#pragma once

#include <stdint.h>
#include <sstream>
#include <type_traits>

#include <ODLpp/DeviceVector.h>
#include <ODLpp/CudaVectorImpl.h>

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
#define X(fun) template <typename Tin, typename Tout> void ufunc##fun##(const CudaVectorImpl<Tin>& in, CudaVectorImpl<Tout>& out) { throw std::domain_error("##fun UFunc not supported with this type"); }
ODLPP_FOR_EACH_UFUNC
#undef X

//Implementations for floats
#define X(fun) template <> void ufunc##fun##(const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out);
ODLPP_FOR_EACH_UFUNC
#undef X