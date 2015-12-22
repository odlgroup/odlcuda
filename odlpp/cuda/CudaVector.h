#pragma once

#include <stdint.h>

// Thrust bug...
#define DEBUG 1
#include <pybind11/pybind11.h>
// Disable deprecated API

#include <iostream>
#define _DEBUG 1
#include <thrust/device_vector.h>

// Numpy
#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <odlpp/cuda/CudaVectorImpl.h>
#include <odlpp/cuda/SliceHelper.h>

namespace py = pybind11;

template <typename T>
CudaVectorImpl<T> fromPointer(uintptr_t ptr, size_t size, ptrdiff_t stride) {
    return CudaVectorImpl<T>(CudaVectorImpl<T>::fromPointer(ptr, size, stride));
}

template <typename T>
CudaVectorImpl<T> getSliceView(CudaVectorImpl<T>& vector,
                               const py::slice index) {
    sliceHelper sh(index, vector.size());
    uintptr_t input_data_begin = vector.dataPtr();
    uintptr_t output_data_begin =
        input_data_begin + vector.stride() * sh.start * sizeof(T);
    if (sh.step < 0) output_data_begin -= sizeof(T);

    return fromPointer<T>(output_data_begin, sh.numel,
                          sh.step * vector.stride());
}

template <typename T>
void copyDeviceToHost(CudaVectorImpl<T>& vector,
                      const py::slice index,
                      py::array& target) {
    sliceHelper sh(index, vector.size());

    if (sh.numel != target.request().count)
        throw std::out_of_range("Size of array does not match slice");

    if (sh.numel > 0) {
        vector.getSliceImpl(*vector._impl, sh.start, sh.stop, sh.step,
                            getDataPtr<T>(target));
    }
}

template <typename T>
py::array getSliceToHost(CudaVectorImpl<T>& vector, const py::slice index) {
    sliceHelper sh(index, vector.size());
    //TODO: remove?
    _import_array();
    if (sh.numel > 0) {
        py::array arr = makeArray<T>(sh.numel);
        copyDeviceToHost<T>(vector, index, arr);
        return arr;
    } else {
        return makeArray<T>(0);
    }
}

template <typename T>
void setSlice(CudaVectorImpl<T>& vector, const py::slice index,
              py::array& arr) {
    sliceHelper sh(index, vector.size());

    if (sh.numel != arr.request().count)
        throw std::out_of_range("Size of array does not match slice");

    if (sh.numel > 0) {
        vector.setSliceImpl(*vector._impl, sh.start, sh.stop, sh.step,
                            getDataPtr<T>(arr), sh.numel);
    }
}

template <typename T>
std::ostream& operator<<(std::ostream& ss, const CudaVectorImpl<T>& v) {
    ss << "CudaVectorImpl<" << typeid(T).name() << ">: ";
    auto outputIter = std::ostream_iterator<T>(ss, " ");
    v.printData(outputIter, std::min<size_t>(100, v.size()));
    return ss;
}

template <typename T>
std::string repr(const CudaVectorImpl<T>& vector) {
    std::stringstream ss;
    ss << vector;
    return ss.str();
}

template <typename T>
py::object dtype(const CudaVectorImpl<T>& v) {
    PyArray_Descr* descr = PyArray_DescrNewFromType(getEnum<T>());
    return py::object(reinterpret_cast<PyObject*>(descr), false);
}

template <typename T>
size_t nbytes(const CudaVectorImpl<T>& vector) {
    return vector.size() * sizeof(T);
}

template <typename T>
size_t itemsize(const CudaVectorImpl<T>& vector) {
    return sizeof(T);
}

template <typename T>
py::tuple shape(const CudaVectorImpl<T>& v) {
    py::tuple result(1);
    result[0] = py::int_(v.size());
    return result;
}
