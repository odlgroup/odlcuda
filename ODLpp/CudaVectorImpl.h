#pragma once

// Disable deprecated API
#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdint.h>
#include <sstream>

#include <thrust/device_vector.h>
#include <numpy/arrayobject.h>
#include <ODLpp/DeviceVector.h>

#include <LCRUtils/python/numpy_utils.h>

using namespace boost::python;

template <typename T>
class CudaVectorImpl {
   public:
    CudaVectorImpl(size_t size);
    CudaVectorImpl(size_t size, T value);
    CudaVectorImpl(size_t size, DeviceVectorPtr<T> impl);

    static CudaVectorImpl<T> fromPointer(uintptr_t ptr, size_t size);

    T getItem(ptrdiff_t index) const;
    void setItem(ptrdiff_t index, T value);

    numeric::array getSlice(const slice index) const;
    void setSlice(const slice index, const numeric::array& arr);

    // numerical methods
    void linComb(T a, const CudaVectorImpl<T>& x, T b,
                 const CudaVectorImpl<T>& y);
    double dist(const CudaVectorImpl<T>& other) const;
    double norm() const;
    double inner(const CudaVectorImpl<T>& v2) const;
    void multiply(const CudaVectorImpl<T>& v1, const CudaVectorImpl<T>& v2);

    // Convenience methods
    CudaVectorImpl<T> copy() const;
    bool allEqual(const CudaVectorImpl<T>& v2) const;

    friend std::ostream& operator<<(std::ostream& ss, const CudaVectorImpl& v) {
        ss << "CudaVectorImpl<" << typeid(T).name() << ">: ";
        auto outputIter = std::ostream_iterator<T>(ss, " ");
        v.printData(outputIter, std::min<int>(100, v._size));
        return ss;
    }

    std::string repr() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    // Implicit conversion to the data container
    operator DeviceVector<T>&();
    operator const DeviceVector<T>&() const;

    // Accessors for data
    uintptr_t dataPtr() const;

    boost::python::object dtype() const {
        PyArray_Descr* descr = PyArray_DescrNewFromType(getEnum<T>());
        return boost::python::object(
            boost::python::handle<>(reinterpret_cast<PyObject*>(descr)));
    }

    size_t size() const { return _size; }

    boost::python::tuple shape() const {
        return boost::python::make_tuple(_size);
    }

   private:
    // Members
    const size_t _size;
    DeviceVectorPtr<T> _impl;

    void validateIndex(ptrdiff_t index) const;
    void printData(std::ostream_iterator<T>& out, int numel) const;
};
