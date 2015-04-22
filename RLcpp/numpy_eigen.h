#pragma once

#include "boost/python/numeric.hpp"
#include <numpy/arrayobject.h>
#include <Python.h>

#include <Eigen/Dense>

#include <stdexcept>
#include <typeinfo>
#include <numeric>

#include <RLcpp/numpy_utils.h>

using namespace boost::python;

template <typename EigenArray>
void verifySize(const EigenSize& size) {
	static const int rowsAtCompile = EigenArray::RowsAtCompileTime;
	static const int colsAtCompile = EigenArray::ColsAtCompileTime;

	if (size.dimension == 1 &&
		((rowsAtCompile != Eigen::Dynamic && rowsAtCompile != 1) &&
		(colsAtCompile != Eigen::Dynamic && colsAtCompile != 1))) {
		throw std::logic_error(("Dimensions not equal expected 2, got " + std::to_string(size.dataCols) + "x" + std::to_string(size.dataRows)).c_str());
	}
	if (size.dimension == 2 &&
		((rowsAtCompile == 1) ||
		(colsAtCompile == 1))) {

		throw std::logic_error((std::to_string(size.dataRows) + " " + std::to_string(size.dataCols)).c_str());
		throw std::logic_error(("Dimensions not equal expected 1, got " + std::to_string(size.dataCols) + "x" + std::to_string(size.dataRows)).c_str());
	}

	if ((rowsAtCompile != Eigen::Dynamic && rowsAtCompile != size.dataRows) ||
		(colsAtCompile != Eigen::Dynamic && colsAtCompile != size.dataCols)) {
		throw std::logic_error(("Sizes not equal, expected " + std::to_string(rowsAtCompile) + "x" + std::to_string(colsAtCompile) +
			" got " + std::to_string(size.dataRows) + "x" + std::to_string(size.dataCols)).c_str());
	}
}

bool iscontiguous(const numeric::array& arr){
	return PyArray_ISCONTIGUOUS(reinterpret_cast<PyArrayObject*>(arr.ptr()));
}

template <typename EigenArray>
bool isPtrCompatible(const numeric::array& numpyArray) {
	typedef typename EigenArray::Scalar Scalar;

	if (!isType<Scalar>(numpyArray))
		return false;

	if (!iscontiguous(numpyArray))
		return false;

	//TODO: use actual order of numpyArray
	if (!EigenArray::IsRowMajor)
		return false;

	return true;
}

template <typename EigenArray>
Eigen::Map<EigenArray> mapInput(numeric::array data) {
	typedef typename EigenArray::Scalar Scalar;

	EigenSize size = getSize(data);
	verifySize<EigenArray>(size);

	return Eigen::Map<EigenArray>(getDataPtr<Scalar>(data), size.dataRows, size.dataCols);
}

template <typename EigenArray>
void copyElements(const EigenSize& size, const object& in, EigenArray& out) {
	typedef typename EigenArray::Scalar Scalar;

	if (size.datadimension == 1) {
		for (size_t i = 0; i < size.dataRows; i++)
			out(i) = extract<Scalar>(in[i]);
	}
	else {
		for (size_t i = 0; i < size.dataRows; i++) {
			for (size_t j = 0; j < size.dataCols; j++)
				out(i, j) = extract<Scalar>(in[i][j]);
		}
	}
}

template <typename EigenArray>
EigenArray copyInput(const object& data) {
	typedef typename EigenArray::Scalar Scalar;
	typedef typename Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorDoubleArray;

	extract<numeric::array> asNumeric(data);
	if (asNumeric.check()) {
		//If data is an array, attempt some efficient methods first

		numeric::array dataArray = asNumeric();

		EigenSize size = getSize(dataArray);
		verifySize<EigenArray>(size);

		EigenArray out(size.dataRows, size.dataCols);

		if (isPtrCompatible<EigenArray>(dataArray)) {
			//Use raw buffers if possible
			auto mapped = mapInput<EigenArray>(dataArray);
			out = mapped;
		}
		else if (isPtrCompatible<RowMajorDoubleArray>(dataArray)) {
			//Default implementation for double numpy array
			auto mapped = mapInput<RowMajorDoubleArray>(dataArray);
			out = mapped.cast<Scalar>(); //If out type does not equal in type, perform a cast. If equal this is assignment.
		}
		else {
			//Slow method if raw buffers unavailable.
			copyElements(size, data, out);
		}

		return out;
	}
	else {
		EigenSize size = getSizeGeneral(data);
		verifySize<EigenArray>(size);

		EigenArray out(size.dataRows, size.dataCols);

		copyElements(size, data, out);

		return out;
	}
}

template <typename EigenArray>
numeric::array copyOutput(const EigenArray& data) {
	typedef typename EigenArray::Scalar Scalar;

	npy_intp dims[2] = { (npy_intp)data.rows(), (npy_intp)data.cols() };

	object obj(handle<>(PyArray_SimpleNew(2, dims, getEnum<Scalar>())));
	numeric::array arr = extract<numeric::array>(obj);

	auto mapped = mapInput<Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(arr); //Numpy uses Row Major storage
	mapped = data;

	return extract<numeric::array>(arr.copy()); //Copy to pass ownership
}
