#pragma once

#include <memory>

// thrust
#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

// ODL
#include <odlcuda/cuda/DeviceVector.h>

// Utils
#include <odl_cpp_utils/cuda/thrustUtils.h>

template <typename T>
class DeviceVector {
   public:
    virtual ~DeviceVector() {}
    virtual T* data() = 0;
    virtual T const* data() const = 0;
    virtual size_t size() const = 0;
    virtual ptrdiff_t stride() const = 0;

    typename strided_range<thrust::device_ptr<T>>::iterator begin() {
        auto begin_data = thrust::device_pointer_cast<T>(data());
        auto end_data = begin_data + size() * stride();
        auto range = make_strided_range(begin_data, end_data, stride());
        return range.begin();
    }
    typename strided_range<thrust::device_ptr<const T>>::iterator begin()
        const {
        auto begin_data = thrust::device_pointer_cast<const T>(data());
        auto end_data = begin_data + size() * stride();
        auto range = make_strided_range(begin_data, end_data, stride());
        return range.begin();
    }

    typename strided_range<thrust::device_ptr<T>>::iterator end() {
        return begin() + size();
    }
    typename strided_range<thrust::device_ptr<const T>>::iterator end() const {
        return begin() + size();
    }

    thrust::device_reference<T> operator[](size_t index) {
        return thrust::device_reference<T>{thrust::device_pointer_cast(data()) +
                                           stride() * index};
    }
    thrust::device_reference<const T> operator[](size_t index) const {
        return thrust::device_reference<T>{thrust::device_pointer_cast(data()) +
                                           stride() * index};
    }
};

template <typename T>
class ThrustDeviceVector : public DeviceVector<T> {
   private:
    thrust::device_vector<T> _data;

   public:
    ThrustDeviceVector(size_t size) : _data(size) {}

    ThrustDeviceVector(size_t size, T value) : _data(size, value) {}

    ThrustDeviceVector(const DeviceVector<T>& other)
        : _data(other.begin(), other.end()) {}

    T* data() override { return thrust::raw_pointer_cast(_data.data()); }
    T const* data() const override {
        return thrust::raw_pointer_cast(_data.data());
    }

    size_t size() const override { return _data.size(); }
    ptrdiff_t stride() const override { return 1; }
};

template <typename T>
class WrapperDeviceVector : public DeviceVector<T> {
   private:
    T* const _data;
    const size_t _size;
    const ptrdiff_t _stride;

   public:
    WrapperDeviceVector(T* data, size_t size, ptrdiff_t stride)
        : _data(data), _size(size), _stride(stride) {}

    T* data() override { return _data; }

    T const* data() const override { return _data; }

    size_t size() const override { return _size; }

    ptrdiff_t stride() const override { return _stride; }
};
