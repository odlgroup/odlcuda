#pragma once

#include <stdint.h>
#include <sstream>
#include <type_traits>

#include <odlcuda/cuda/TypeMacro.h>
#include <odlcuda/cuda/DeviceVector.h>

// The scalar type used for multiplication
template <typename T, typename Enable = void>
struct CudaVectorTraits {
    using Scalar = T;
    using Float = double;
    using RealFloat = double;
};

template <typename T>
struct CudaVectorTraits<
    T, typename std::enable_if<std::is_integral<T>::value>::type> {
    using Scalar = typename std::make_signed<T>::type;
    using Float = double;
    using RealFloat = double;
};

template <typename T>
struct CudaVectorTraits<
    T, typename std::enable_if<std::is_same<T, float>::value>::type> {
    using Scalar = float;
    using Float = float;
    using RealFloat = float;
};

template <typename T>
class CudaVectorImpl {
   public:
    using Scalar = typename CudaVectorTraits<T>::Scalar;
    using Float = typename CudaVectorTraits<T>::Float;
    using RealFloat = typename CudaVectorTraits<T>::RealFloat;

    CudaVectorImpl(size_t size);
    CudaVectorImpl(size_t size, T value);
    CudaVectorImpl(DeviceVectorPtr<T> impl);

    static DeviceVectorPtr<T> fromPointer(uintptr_t ptr, size_t size,
                                          ptrdiff_t stride);

    T getItem(ptrdiff_t index) const;
    void setItem(ptrdiff_t index, T value);

    // numerical methods
    void linComb(Scalar a, const CudaVectorImpl<T>& x, Scalar b,
                 const CudaVectorImpl<T>& y);
    void multiply(const CudaVectorImpl<T>& v1, const CudaVectorImpl<T>& v2);
    void divide(const CudaVectorImpl<T>& v1, const CudaVectorImpl<T>& v2);

    // dist
    RealFloat dist(const CudaVectorImpl<T>& other) const;
    RealFloat dist_power(const CudaVectorImpl<T>& other, RealFloat power) const;
    RealFloat dist_weight(const CudaVectorImpl<T>& other, RealFloat power,
                          const CudaVectorImpl<RealFloat>& weight) const;

    // norm
    RealFloat norm() const;
    RealFloat norm_power(RealFloat power) const;
    RealFloat norm_weight(RealFloat power,
                          const CudaVectorImpl<RealFloat>& weight) const;

    // inner
    Float inner(const CudaVectorImpl<T>& v2) const;
    Float inner_weight(const CudaVectorImpl<T>& v2,
                       const CudaVectorImpl<RealFloat>& weight) const;

    // Convenience methods
    CudaVectorImpl<T> copy() const;
    bool allEqual(const CudaVectorImpl<T>& v2) const;
    void fill(T value);

    // Implicit conversion to the data container
    operator DeviceVector<T>&();
    operator const DeviceVector<T>&() const;

    // Accessors for data
    uintptr_t dataPtr() const;
    ptrdiff_t stride() const;
    size_t size() const;

    // Raw copy
    void getSliceImpl(const DeviceVector<T>& v1, size_t start, size_t stop,
                      ptrdiff_t step, T* host_target) const;
    void setSliceImpl(DeviceVector<T>& v1, size_t start, size_t stop,
                      ptrdiff_t step, const T* host_source, size_t num);

    // Members
    DeviceVectorPtr<T> _impl;

    void validateIndex(ptrdiff_t index) const;
    // Copy to ostream
    void printData(std::ostream_iterator<T>& out, size_t numel) const;
};
