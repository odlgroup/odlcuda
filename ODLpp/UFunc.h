#pragma once

#include <stdint.h>
#include <sstream>
#include <type_traits>

#include <ODLpp/DeviceVector.h>
#include <ODLpp/CudaVectorImpl.h>

// clang-format off

#define ODLPP_FOR_EACH_UFUNC \
    X(sin) \
    X(cos) \
    X(asin) \
    X(acos) \
    X(log) \
    X(exp)

#define ODLPP_FOR_EACH_FLOAT_UFUNC \
    X(sin, sinf) \
    X(cos, cosf) \
    X(asin, asinf) \
    X(acos, acosf) \
    X(log, logf) \
    X(exp, expf)

template <typename Tin, typename Tout>
struct UFunc {
#define X(fun, impl) void fun##(const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out) {throw std::domain_error("this UFunc not supported with this type");}
    ODLPP_FOR_EACH_UFUNC
#undef X
};

template <>
struct UFunc<float, float> {
#define X(fun, impl) void fun##(const CudaVectorImpl<float>& in, CudaVectorImpl<float>& out);
    ODLPP_FOR_EACH_FLOAT_UFUNC
#undef X
};