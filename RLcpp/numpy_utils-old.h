#pragma once

// Disable deprecated API
#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "boost/python/numeric.hpp"
#include <stdexcept>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <typeinfo>
#include <numeric>

using namespace boost::python;

template <typename T>
int getEnum();

#define makeDefinition(TYPE, NAME) \
	template <>                    \
	int getEnum<TYPE>() { return NAME; }
makeDefinition(float, NPY_FLOAT);
makeDefinition(double, NPY_DOUBLE);
makeDefinition(long double, NPY_LONGDOUBLE);
makeDefinition(bool, NPY_BOOL);
makeDefinition(char, NPY_BYTE);
makeDefinition(unsigned char, NPY_UBYTE);
makeDefinition(short, NPY_SHORT);
makeDefinition(unsigned short, NPY_USHORT);
makeDefinition(int, NPY_INT);
makeDefinition(unsigned int, NPY_UINT);
makeDefinition(long, NPY_LONG);
makeDefinition(unsigned long, NPY_ULONG);
makeDefinition(long long, NPY_LONGLONG);
makeDefinition(unsigned long long, NPY_ULONGLONG);
#undef makeDefinition

template <typename T>
bool isType(const numeric::array& data)
{
	PyArrayObject* a = (PyArrayObject*)data.ptr();

	if (a == NULL)
		return false;

	return PyArray_TYPE(a) == getEnum<T>();
}

struct EigenSize {
	size_t dataRows, dataCols, dimension, datadimension;
};

EigenSize getSize(const numeric::array& data) {
	const tuple& shape = extract<tuple>(data.attr("shape"));
	const size_t datadimension = len(shape);
	size_t dimension = datadimension;

	size_t dataRows, dataCols;
	if (datadimension == 1) {
		dataRows = extract<int>(shape[0]);
		dataCols = 1;
	}
	else if (datadimension == 2) {
		dataRows = extract<int>(shape[0]);
		dataCols = extract<int>(shape[1]);

		if (dataRows == 1 || dataCols == 1)
			dimension = 1;
	}
	else
		throw std::logic_error("Dimension is not 1 or 2");

	return{ dataRows, dataCols, dimension, datadimension };
}

EigenSize getSizeGeneral(const object& data) {
	size_t dataRows = 1;
	size_t dataCols = 1;
	size_t dimension = 1;
	size_t dataDimension = 1;

	try {
		dataRows = len(data);
		if (PyObject_HasAttrString(object(data[0]).ptr(), "len")) {
			dataCols = len(data[0]);
			dataDimension = 2;

			//TODO check that all others are equal
		}
	}
	catch (const error_already_set&) {
		throw std::logic_error("Data is not of array type");
	}

	if (dataRows > 1 && dataCols > 1)
		dimension = 2;

	return{ dataRows, dataCols, dimension, dataDimension };
}

template <typename T>
numeric::array makeArray(size_t size)
{
	npy_intp dims[1] = { size };
	object obj(handle<>(PyArray_SimpleNew(1, dims, getEnum<T>())));
	numeric::array arr = extract<numeric::array>(obj);
	return arr;
}

template <typename T>
numeric::array makeArray(size_t size1, size_t size2)
{
	npy_intp dims[2] = { size1, size2 };
	object obj(handle<>(PyArray_SimpleNew(2, dims, getEnum<T>())));
	numeric::array arr = extract<numeric::array>(obj);
	return arr;
}

template <typename T>
T* getDataPtr(const numeric::array& data) {
	PyArrayObject* a = (PyArrayObject*)data.ptr();

	if (a == NULL)
		throw std::logic_error("Could not get NP array.");

	//Check that type is correct
	if (PyArray_TYPE(a) != getEnum<T>())
		throw std::logic_error(("Expected element type " + std::string(typeid(T).name()) + " " + PyArray_DESCR(a)->type).c_str());

	T* p = (T*)PyArray_DATA(a);

	return p;
}
