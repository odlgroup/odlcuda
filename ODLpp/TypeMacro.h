#pragma once

#if 0
//full compile
#define ODLPP_FOR_EACH_TYPE \
    X(int8_t, "CudaVectorInt8") \
    X(int16_t, "CudaVectorInt16") \
    X(int32_t, "CudaVectorInt32") \
    X(int64_t, "CudaVectorInt64") \
    X(uint8_t, "CudaVectorUInt8") \
    X(uint16_t, "CudaVectorUInt16") \
    X(uint32_t, "CudaVectorUInt32") \
    X(uint64_t, "CudaVectorUInt64") \
    X(float, "CudaVectorFloat32") \
    X(double, "CudaVectorFloat64")
#else

//debug type compile
#define ODLPP_FOR_EACH_TYPE \
    X(uint8_t, "CudaVectorUInt8") \
    X(float, "CudaVectorFloat32")
#endif