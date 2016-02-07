#include <algorithm>
#include <memory>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// thrust
#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/adjacent_difference.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <iostream>

// ODL
#include <odlpp/cuda/DeviceVectorImpl.h>
#include <odlpp/cuda/CudaVectorImpl.h>
#include <odlpp/cuda/TypeMacro.h>

// Utils
#include <odl_cpp_utils/cuda/thrustUtils.h>

template <typename I1, typename I2>
void stridedGetImpl(I1 fromBegin, I1 fromEnd, I2 toBegin, ptrdiff_t step) {
    if (step == 1) {
        thrust::copy(fromBegin, fromEnd, toBegin);
    } else {
        auto iter = make_strided_range(fromBegin, fromEnd, step);
        thrust::copy(iter.begin(), iter.end(), toBegin);
    }
}

template <typename I1, typename I2>
void stridedSetImpl(I1 fromBegin, I1 fromEnd, I2 toBegin, I2 toEnd,
                    ptrdiff_t step) {
    if (step == 1) {
        thrust::copy(fromBegin, fromEnd, toBegin);
    } else {
        auto iter = make_strided_range(toBegin, toEnd, step);
        thrust::copy(fromBegin, fromEnd, iter.begin());
    }
}
template <typename T>
void CudaVectorImpl<T>::getSliceImpl(const DeviceVector<T>& v1, size_t start,
                                     size_t stop, ptrdiff_t step,
                                     T* target) const {
    if (step > 0) {
        stridedGetImpl(v1.begin() + start, v1.begin() + stop, target, step);
    } else {
        auto reversedBegin = thrust::make_reverse_iterator(v1.begin() + start);
        auto reversedEnd = thrust::make_reverse_iterator(v1.begin() + stop);

        stridedGetImpl(reversedBegin, reversedEnd, target, -step);
    }
}
template <typename T>
void CudaVectorImpl<T>::setSliceImpl(DeviceVector<T>& v1, size_t start,
                                     size_t stop, ptrdiff_t step,
                                     const T* source, size_t num) {
    if (step > 0) {
        stridedSetImpl(source, source + num, v1.begin() + start,
                       v1.begin() + stop, step);
    } else {
        auto reversedBegin = thrust::make_reverse_iterator(v1.begin() + start);
        auto reversedEnd = thrust::make_reverse_iterator(v1.begin() + stop);

        stridedSetImpl(source, source + num, reversedBegin, reversedEnd, -step);
    }
}
template <typename T>
CudaVectorImpl<T>::CudaVectorImpl(size_t size)
    : _impl(std::make_shared<ThrustDeviceVector<T>>(size)) {}

template <typename T>
CudaVectorImpl<T>::CudaVectorImpl(size_t size, T value)
    : _impl(std::make_shared<ThrustDeviceVector<T>>(size, value)) {}

template <typename T>
CudaVectorImpl<T>::CudaVectorImpl(DeviceVectorPtr<T> impl)
    : _impl(impl) {}

template <typename T>
DeviceVectorPtr<T> CudaVectorImpl<T>::fromPointer(
    uintptr_t ptr, size_t size, ptrdiff_t stride) {
    return std::make_shared<WrapperDeviceVector<T>>(reinterpret_cast<T*>(ptr),
                                                    size, stride);
}

template <typename T>
void validateIndex(const CudaVectorImpl<T>& vector, ptrdiff_t index) {
	if (index < 0 || index >= static_cast<ptrdiff_t>(vector.size()))
		throw std::out_of_range("index out of range");
}

template <typename T>
T CudaVectorImpl<T>::getItem(ptrdiff_t index) const {
	validateIndex(*this, index);
    return _impl->operator[](index);
}

template <typename T>
void CudaVectorImpl<T>::setItem(ptrdiff_t index, T value) {
	validateIndex(*this, index);
    _impl->operator[](index) = value;
}

template <typename T, typename Scalar>
void linCombImpl(DeviceVector<T>& z, Scalar a, const DeviceVector<T>& x,
                 Scalar b, const DeviceVector<T>& y) {
    namespace ph = thrust::placeholders;

#if 1 // Efficient
    if (a == Scalar(0)) {
        if (b == Scalar(0)) { // z = 0
            thrust::fill(z.begin(), z.end(), T(0));
        } else if (b == Scalar(1)) { // z = y
            thrust::copy(y.begin(), y.end(), z.begin());
        } else if (b == -Scalar(1)) { // y = -y
            thrust::transform(y.begin(), y.end(), z.begin(), -ph::_1);
        } else { // y = b*y
            thrust::transform(y.begin(), y.end(), z.begin(), b * ph::_1);
        }
    } else if (a == Scalar(1)) {
        if (b == Scalar(0)) { // z = x
            thrust::copy(x.begin(), x.end(), z.begin());
        } else if (b == Scalar(1)) { // z = x+y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              thrust::plus<T>());
        } else if (b == -Scalar(1)) { // z = x-y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              thrust::minus<T>());
        } else { // z = x + b*y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              ph::_1 + b * ph::_2);
        }
    } else if (a == -Scalar(1)) {
        if (b == Scalar(0)) { // z = -x
            thrust::transform(x.begin(), x.end(), z.begin(), -ph::_1);
        } else if (b == Scalar(1)) { // z = -x+y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              -ph::_1 + ph::_2);
        } else if (b == -Scalar(1)) { // z = -x-y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              -ph::_1 - ph::_2);
        } else { // z = -x + b*y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              -ph::_1 + b * ph::_2);
        }
    } else {
        if (b == Scalar(0)) { // z = a*x
            thrust::transform(x.begin(), x.end(), z.begin(), a * ph::_1);
        } else if (b == Scalar(1)) { // z = a*x+y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              a * ph::_1 + ph::_2);
        } else if (b == -Scalar(1)) { // z = a*x-y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              a * ph::_1 - ph::_2);
        } else { // z = a*x + b*y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                              a * ph::_1 + b * ph::_2);
        }
    }
#else // Basic
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                      a * ph::_1 + b * ph::_2);
#endif
}

template <typename T>
void CudaVectorImpl<T>::linComb(Scalar a, const CudaVectorImpl<T>& x, Scalar b,
                                const CudaVectorImpl<T>& y) {
    linCombImpl(*this->_impl, a, *x._impl, b, *y._impl);
}

// dist

template <typename T>
struct DistanceFunctor {
    using RealFloat = typename CudaVectorImpl<T>::RealFloat;

    __host__ __device__ RealFloat
    operator()(const thrust::tuple<T, T>& f) const {
        return static_cast<RealFloat>((thrust::get<0>(f) - thrust::get<1>(f)) *
                                      (thrust::get<0>(f) - thrust::get<1>(f)));
    }
};
template <typename T>
CudaVectorImpl<T>::RealFloat CudaVectorImpl<T>::dist(
    const CudaVectorImpl<T>& other) const {
    auto first = thrust::make_zip_iterator(thrust::make_tuple(this->_impl->begin(), other._impl->begin()));
    auto last = first + this->size();

    return sqrt(thrust::transform_reduce(
        first, last, DistanceFunctor<T>(), CudaVectorImpl<T>::RealFloat(0),
        thrust::plus<CudaVectorImpl<T>::RealFloat>()));
}

template <typename T>
struct DistanceFunctorPower {
    using RealFloat = typename CudaVectorImpl<T>::RealFloat;

    RealFloat _power;

    DistanceFunctorPower(RealFloat power) : _power(power) {}

    __host__ __device__ RealFloat
    operator()(const thrust::tuple<T, T>& f) const {
        return pow(static_cast<RealFloat>(
                       fabsf(thrust::get<0>(f) - thrust::get<1>(f))),
                   _power);
    }
};
template <typename T>
CudaVectorImpl<T>::RealFloat CudaVectorImpl<T>::dist_power(const CudaVectorImpl<T>& other, CudaVectorImpl<T>::RealFloat power) const {
    auto first = thrust::make_zip_iterator(thrust::make_tuple(this->_impl->begin(), other._impl->begin()));
    auto last = first + this->size();
    auto dist_func = DistanceFunctorPower<T>(power);

    return pow(thrust::transform_reduce(
                   first, last, dist_func, CudaVectorImpl<T>::RealFloat(0),
                   thrust::plus<CudaVectorImpl<T>::RealFloat>()),
               1.0f / power);
}

template <typename T>
struct DistanceFunctorWeighted {
    using RealFloat = typename CudaVectorImpl<T>::RealFloat;

    RealFloat _power;

    DistanceFunctorWeighted(RealFloat power) : _power(power) {}

    __host__ __device__ RealFloat
    operator()(const thrust::tuple<T, T, RealFloat>& f) const {
        return thrust::get<2>(f) *
               pow(static_cast<RealFloat>(
                       fabsf(thrust::get<0>(f) - thrust::get<1>(f))),
                   _power);
    }
};
template <typename T>
CudaVectorImpl<T>::RealFloat CudaVectorImpl<T>::dist_weight(
    const CudaVectorImpl<T>& other, CudaVectorImpl<T>::RealFloat power,
    const CudaVectorImpl<CudaVectorImpl<T>::RealFloat>& weight) const {
    auto first = thrust::make_zip_iterator(thrust::make_tuple(
        this->_impl->begin(), other._impl->begin(), weight._impl->begin()));
    auto last = first + this->size();
    auto dist_func = DistanceFunctorWeighted<T>(power);
    return pow(thrust::transform_reduce(
                   first, last, dist_func, CudaVectorImpl<T>::RealFloat(0),
                   thrust::plus<CudaVectorImpl<T>::RealFloat>()),
               1.0f / power);
}

// norm

template <typename T>
CudaVectorImpl<T>::RealFloat CudaVectorImpl<T>::norm() const {
    namespace ph = thrust::placeholders;
    return sqrt(
        thrust::transform_reduce(this->_impl->begin(), this->_impl->end(),
                                 ph::_1 * ph::_1, CudaVectorImpl<T>::Float(0),
                                 thrust::plus<CudaVectorImpl<T>::RealFloat>()));
}

template <typename T>
struct NormFunctorPower {
    using RealFloat = typename CudaVectorImpl<T>::RealFloat;

    RealFloat _power;

    NormFunctorPower(RealFloat power) : _power(power) {}

    __host__ __device__ RealFloat operator()(T f) const {
        return pow(static_cast<RealFloat>(fabsf(f)), _power);
    }
};
template <typename T>
CudaVectorImpl<T>::RealFloat CudaVectorImpl<T>::norm_power(
    CudaVectorImpl<T>::RealFloat power) const {
    auto norm_func = NormFunctorPower<T>(power);
    return pow(
        thrust::transform_reduce(this->_impl->begin(), this->_impl->end(),
                                 norm_func, CudaVectorImpl<T>::RealFloat(0),
                                 thrust::plus<CudaVectorImpl<T>::RealFloat>()),
        1.0f / power);
}

template <typename T>
struct NormFunctorWeighted {
    using RealFloat = typename CudaVectorImpl<T>::RealFloat;

    RealFloat _power;

    NormFunctorWeighted(RealFloat power) : _power(power) {}

    __host__ __device__ RealFloat
    operator()(const thrust::tuple<T, RealFloat>& f) const {
        return thrust::get<1>(f) *
               pow(static_cast<RealFloat>(fabsf(thrust::get<0>(f))), _power);
    }
};
template <typename T>
CudaVectorImpl<T>::RealFloat CudaVectorImpl<T>::norm_weight(
    CudaVectorImpl<T>::RealFloat power,
    const CudaVectorImpl<CudaVectorImpl<T>::RealFloat>& weight) const {
    auto first = thrust::make_zip_iterator(
        thrust::make_tuple(this->_impl->begin(), weight._impl->begin()));
    auto last = first + this->size();
    auto norm_func = NormFunctorWeighted<T>(power);
    return pow(thrust::transform_reduce(
                   first, last, norm_func, CudaVectorImpl<T>::RealFloat(0),
                   thrust::plus<CudaVectorImpl<T>::RealFloat>()),
               1.0f / power);
}

template <typename T>
CudaVectorImpl<T>::Float CudaVectorImpl<T>::inner(
    const CudaVectorImpl<T>& other) const {
    return thrust::inner_product(this->_impl->begin(), this->_impl->end(),
                                 other._impl->begin(),
                                 CudaVectorImpl<T>::Float(0));
}

template <typename T>
struct InnerFunctorWeighted {
    using Float = typename CudaVectorImpl<T>::Float;
    using RealFloat = typename CudaVectorImpl<T>::RealFloat;

    __host__ __device__ RealFloat
    operator()(const thrust::tuple<T, T, RealFloat>& f) const {
        return thrust::get<0>(f) * thrust::get<1>(f) * thrust::get<2>(f);
    }
};
template <typename T>
CudaVectorImpl<T>::Float CudaVectorImpl<T>::inner_weight(
    const CudaVectorImpl<T>& other,
    const CudaVectorImpl<CudaVectorImpl<T>::RealFloat>& weight) const {
    auto first = thrust::make_zip_iterator(thrust::make_tuple(
        this->_impl->begin(), other._impl->begin(), weight._impl->begin()));
    auto last = first + this->size();
    auto inner_func = InnerFunctorWeighted<T>();
    return thrust::transform_reduce(first, last, inner_func,
                                    CudaVectorImpl<T>::Float(0),
                                    thrust::plus<CudaVectorImpl<T>::Float>());
}

template <typename T>
void CudaVectorImpl<T>::multiply(const CudaVectorImpl<T>& x,
                                 const CudaVectorImpl<T>& y) {
    thrust::transform(x._impl->begin(), x._impl->end(), y._impl->begin(),
                      this->_impl->begin(), thrust::multiplies<T>());
}

template <typename T>
void CudaVectorImpl<T>::divide(const CudaVectorImpl<T>& x,
                               const CudaVectorImpl<T>& y) {
    thrust::transform(x._impl->begin(), x._impl->end(), y._impl->begin(),
                      this->_impl->begin(), thrust::divides<T>());
}

template <typename T>
CudaVectorImpl<T> CudaVectorImpl<T>::copy() const {
    DeviceVectorPtr<T> data_cpy =
        std::make_shared<ThrustDeviceVector<T>>(*_impl);
    return CudaVectorImpl<T>(data_cpy);
}

template <typename T>
bool CudaVectorImpl<T>::allEqual(const CudaVectorImpl<T>& other) const {
    return thrust::equal(this->_impl->begin(), this->_impl->end(),
                         other._impl->begin());
}

template <typename T>
void CudaVectorImpl<T>::fill(T value) {
    thrust::fill(this->_impl->begin(), this->_impl->end(), value);
}

template <typename T>
void CudaVectorImpl<T>::printData(std::ostream_iterator<T>& out,
                                  size_t numel) const {
    thrust::copy(this->_impl->begin(), this->_impl->begin() + numel, out);
}

template <typename T>
CudaVectorImpl<T>::operator DeviceVector<T>&() {
    return *this->_impl;
}

template <typename T>
CudaVectorImpl<T>::operator const DeviceVector<T>&() const {
    return *this->_impl;
}

template <typename T>
uintptr_t CudaVectorImpl<T>::dataPtr() const {
    return reinterpret_cast<uintptr_t>(_impl->data());
}

template <typename T>
ptrdiff_t CudaVectorImpl<T>::stride() const {
    return _impl->stride();
}

template <typename T>
size_t CudaVectorImpl<T>::size() const {
    return _impl->size();
}

// Instantiate the methods for each type
#define X(type, name) template struct CudaVectorImpl<type>;
ODL_CUDA_FOR_EACH_TYPE
#undef X
