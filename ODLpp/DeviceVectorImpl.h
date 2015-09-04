#pragma once

#include <memory>

// thrust
#include <LCRUtils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <LCRUtils/cuda/enableThrustWarnings.h>

// ODL
#include <ODLpp/DeviceVector.h>

// Utils
#include <LCRUtils/cuda/thrustUtils.h>

template <typename T>
class DeviceVector {
   public:
    virtual ~DeviceVector() {}
    virtual T* data() = 0;
    virtual T const* data() const = 0;
    virtual size_t size() const = 0;

    thrust::device_ptr<T> begin() {
        return thrust::device_pointer_cast<T>(data());
    }
    thrust::device_ptr<const T> begin() const {
        return thrust::device_pointer_cast<const T>(data());
    }

    thrust::device_ptr<T> end() { return begin() + size(); }
    thrust::device_ptr<const T> end() const { return begin() + size(); }

    thrust::device_reference<T> operator[](size_t index) {
        return thrust::device_reference<T>{begin() + index};
    }
    thrust::device_reference<const T> operator[](size_t index) const {
        return thrust::device_reference<const T>{begin() + index};
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
};

template <typename T>
class WrapperDeviceVector : public DeviceVector<T> {
   private:
    T* const _data;
    const size_t _size;

   public:
    WrapperDeviceVector(T* data, size_t size) : _data(data), _size(size) {}

    T* data() override { return _data; }

    T const* data() const override { return _data; }

    size_t size() const override { return _size; }
};
