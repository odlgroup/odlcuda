#pragma once

// Disable deprecated API
#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdint.h>

// Thrust bug...
#define DEBUG 1
#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <iostream>
#define _DEBUG 1
#include <thrust/device_vector.h>
#include <numpy/arrayobject.h>

#include <ODLpp/CudaVectorImpl.h>
#include <ODLpp/SliceHelper.h>

using namespace boost::python;

template <typename T>
CudaVectorImpl<T> fromPointer(uintptr_t ptr, size_t size) {
    return CudaVectorImpl<T>(size, CudaVectorImpl<T>::fromPointer(ptr, size));
}

template <typename T>
boost::python::numeric::array getSlice(CudaVectorImpl<T>& vector,
                                       const boost::python::slice index) {
    sliceHelper sh(index, vector._size);
    _import_array();
    if (sh.numel > 0) {
        numeric::array arr = makeArray<T>(sh.numel);
        vector.getSliceImpl(*vector._impl, sh.start, sh.stop, sh.step, getDataPtr<T>(arr));
        return arr;
    } else {
        return makeArray<T>(0);
    }
}

template <typename T>
void setSlice(CudaVectorImpl<T>& vector,
              const boost::python::slice index,
              const boost::python::numeric::array& arr) {
    sliceHelper sh(index, vector._size);

    if (sh.numel != len(arr))
        throw std::out_of_range("Size of array does not match slice");

    if (sh.numel > 0) {
        vector.setSliceImpl(*vector._impl, sh.start, sh.stop, sh.step, getDataPtr<T>(arr),
                            sh.numel);
    }
}

template <typename T>
std::ostream& operator<<(std::ostream& ss, const CudaVectorImpl<T>& v) {
    ss << "CudaVectorImpl<" << typeid(T).name() << ">: ";
    auto outputIter = std::ostream_iterator<T>(ss, " ");
    v.printData(outputIter, std::min<int>(100, v._size));
    return ss;
}

template <typename T>
std::string repr(const CudaVectorImpl<T>& vector) {
    std::stringstream ss;
    ss << vector;
    return ss.str();
}

template <typename T>
boost::python::object dtype(const CudaVectorImpl<T>& v) {
    PyArray_Descr* descr = PyArray_DescrNewFromType(getEnum<T>());
    return boost::python::object(
        boost::python::handle<>(reinterpret_cast<PyObject*>(descr)));
}

template <typename T>
boost::python::tuple shape(const CudaVectorImpl<T>& v) {
    return boost::python::make_tuple(v._size);
}