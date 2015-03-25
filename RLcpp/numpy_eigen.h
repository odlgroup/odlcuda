#pragma once

#include "boost/python/numeric.hpp"
#include <numpy/arrayobject.h>
#include <Python.h>

#include <Eigen/Dense>

#include <exception>
#include <typeinfo>
#include <numeric>

#include <RLcpp/numpy_utils.h>

using namespace boost::python;

template <typename EigenArray>
void verifySize(const EigenSize& size) {
    static const int rowsAtCompile = internal::traits<EigenArray>::RowsAtCompileTime;
    static const int colsAtCompile = internal::traits<EigenArray>::ColsAtCompileTime;

    if (size.dimension == 1 &&
        ((rowsAtCompile != Eigen::Dynamic && rowsAtCompile != 1) &&
         (colsAtCompile != Eigen::Dynamic && colsAtCompile != 1))) {
        throw std::exception(("Dimensions not equal expected 2, got " + std::to_string(size.dataCols) + "x" + std::to_string(size.dataRows)).c_str());
    }
    if (size.dimension == 2 &&
        ((rowsAtCompile == 1) ||
         (colsAtCompile == 1))) {

        throw std::exception((std::to_string(size.dataRows) + " " + std::to_string(size.dataCols)).c_str());
        throw std::exception(("Dimensions not equal expected 1, got " + std::to_string(size.dataCols) + "x" + std::to_string(size.dataRows)).c_str());
    }

    if ((rowsAtCompile != Eigen::Dynamic && rowsAtCompile != size.dataRows) ||
        (colsAtCompile != Eigen::Dynamic && colsAtCompile != size.dataCols)) {
        throw std::exception(("Sizes not equal, expected " + std::to_string(rowsAtCompile) + "x" + std::to_string(colsAtCompile) +
                              " got " + std::to_string(size.dataRows) + "x" + std::to_string(size.dataCols)).c_str());
    }
}

template <typename EigenArray>
EigenArray copyInputNP(const numeric::array& data) {
    typedef internal::traits<EigenArray>::Scalar Scalar;

    EigenSize size = getSize(data);
    verifySize<EigenArray>(size);

    //TODO verify size
    EigenArray out(size.dataRows, size.dataCols);
    if (size.datadimension == 1) {
        for (size_t i = 0; i < size.dataRows; i++)
            out(i) = extract<Scalar>(data[i]);
    } else {
        for (size_t i = 0; i < size.dataRows; i++)
            for (size_t j = 0; j < size.dataCols; j++)
                out(i, j) = extract<Scalar>(data[i][j]);
    }

    return out;
}

template <typename EigenArray>
EigenArray copyInput(const object& data) {
    typedef internal::traits<EigenArray>::Scalar Scalar;

    EigenSize size;

    extract<numeric::array> asNumeric(data);
    if (asNumeric.check())
        size = getSize(asNumeric());
    else
        size = getSizeGeneral(data);

    verifySize<EigenArray>(size);

    EigenArray out(size.dataRows, size.dataCols);
    if (size.datadimension == 1) {
        for (size_t i = 0; i < size.dataRows; i++)
            out(i) = extract<Scalar>(data[i]);
    } else {
        for (size_t i = 0; i < size.dataRows; i++)
            for (size_t j = 0; j < size.dataCols; j++)
                out(i, j) = extract<Scalar>(data[i][j]);
    }

    return out;
}

template <typename EigenArray>
Eigen::Map<EigenArray> mapInput(numeric::array data) {
    typedef internal::traits<EigenArray>::Scalar Scalar;

    EigenSize size = getSize(data);
    verifySize<EigenArray>(size);

    Scalar* p = getDataPtr<Scalar>(data);
    return Map<EigenArray>(p, size.dataRows, size.dataCols);
}

template <typename EigenArray>
numeric::array copyOutput(const EigenArray& data) {
    typedef internal::traits<EigenArray>::Scalar Scalar;

    npy_intp dims[2] = {(npy_intp)data.rows(), (npy_intp)data.cols()};

    object obj(handle<>(PyArray_SimpleNew(2, dims, getEnum<Scalar>())));
    numeric::array arr = extract<numeric::array>(obj);

    auto mapped = mapInput<ArrayXXd>(arr);
    mapped = data;

    return arr;
}