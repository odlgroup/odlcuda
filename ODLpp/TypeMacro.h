#pragma once

#include <stdint.h>
#include <complex>
#include <thrust/complex.h>

//We use X macros to duplicate the types properly

#define ODLPP_FOR_EACH_TYPE       \
    X(float, "CudaVectorFloat")   \
    X(double, "CudaVectorDouble") \
    X(int8_t, "CudaVectorInt8")   \
    X(int16_t, "CudaVectorInt16") \
    X(int32_t, "CudaVectorInt32") \
    X(int64_t, "CudaVectorInt64") \
    X(uint8_t, "CudaVectorUInt8")   \
    X(uint16_t, "CudaVectorUInt16") \
    X(uint32_t, "CudaVectorUInt32") \
    X(uint64_t, "CudaVectorUInt64")