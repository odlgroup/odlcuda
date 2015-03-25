#pragma once
//Thrust bug...
#define DEBUG 1
#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <iostream>
#define _DEBUG 1
#include <thrust/device_vector.h>
#include <numpy/arrayobject.h>
#include <RLcpp/numpy_utils.h>

using namespace boost::python;

//Use a special allocator that does not automatically initialize the vector for efficiency
template <typename T>
struct uninitialized_allocator : thrust::device_malloc_allocator<T> {};

typedef std::shared_ptr<thrust::device_vector<float, uninitialized_allocator<float>>> device_vector_ptr;

//Externally (CUDA) compiled
extern device_vector_ptr makeThrustVector(size_t size);
extern device_vector_ptr makeThrustVector(size_t size, float value);
extern void linCombImpl(float a, const device_vector_ptr& v1, float b, device_vector_ptr& v2);
extern void copyHostToDevice(double* source, device_vector_ptr& target);
extern void copyDeviceToHost(const device_vector_ptr& source, double* target);
extern void copyDeviceToDevice(const device_vector_ptr& source, device_vector_ptr& target);
extern float innerProduct(const device_vector_ptr& v1, const device_vector_ptr& v2);
extern void printData(const device_vector_ptr& v1, std::ostream_iterator<float>& out);
extern float getItemImpl(const device_vector_ptr& v1, int index);
extern void setItemImpl(device_vector_ptr& v1, int index, float value);
extern void getSliceImpl(const device_vector_ptr& v1, int begin, int end, int step, double* target);
extern void setSliceImpl(const device_vector_ptr& v1, int begin, int end, int step, double* source, int num);
extern void convImpl(const device_vector_ptr& source, const device_vector_ptr& kernel, device_vector_ptr& target);

struct sliceHelper {
	//TODO move to python?
    sliceHelper(const slice& index, int n) {
        extract<int> startIn(index.start());
        if (startIn.check())
            start = startIn();
        else
            start = 0;

        extract<int> stopIn(index.stop());
        if (stopIn.check())
            stop = stopIn();
        else
            stop = n;

        extract<int> stepIn(index.step());
        if (stepIn.check())
            step = stepIn();
        else
            step = 1;

        if (step > 0) {
            numel = 1 + (stop - start - 1) / step;
        } else //Handle negative steps
        {
            if (!startIn.check())
                start = n - 1;

            if (!stopIn.check())
                stop = -1;

            numel = 1 + std::abs(start - stop - 1) / std::abs(step);

			stop += 2; //Document (something to do with iterators checking while start != end)
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const sliceHelper& sh) {
        return ss << sh.start << " " << sh.stop << " " << sh.step << " " << sh.numel;
    }
    int start, stop, step, numel;
};

class CudaRNVectorImpl {
  public:
    CudaRNVectorImpl(size_t size)
        : _size(size),
          _impl(makeThrustVector(size)) {
    }

    CudaRNVectorImpl(size_t size, float value)
        : _size(size),
          _impl(makeThrustVector(size, value)) {
    }

    CudaRNVectorImpl(const numeric::array& data)
        : _size(len(data)),
          _impl(makeThrustVector(len(data))) {
        assert(this->_size == len(data));
        copyHostToDevice(getDataPtr<double>(data), this->_impl);
    }

    void assign(const CudaRNVectorImpl& other) {
        assert(this->_size == other._size);
        copyDeviceToDevice(other._impl, this->_impl);
    }

    numeric::array copyToHost() const {
		numeric::array arr = makeArray<double>(this->_size);
        copyDeviceToHost(_impl, getDataPtr<double>(arr));
        return arr;
    }

    float getItem(int index) const {
        return getItemImpl(_impl, index);
    }

    void setItem(int index, float value) {
        setItemImpl(_impl, index, value);
    }



    numeric::array getSlice(const slice index) const {
        sliceHelper sh(index, _size);

        if (sh.numel > 0) {
            numeric::array arr = makeArray<double>(sh.numel);
            getSliceImpl(_impl, sh.start, sh.stop, sh.step, getDataPtr<double>(arr));
            return arr;
        } else {
            return makeArray<double>(0);
        }
    }

    void setSlice(const slice index, const numeric::array& arr) {
        sliceHelper sh(index, _size);
        assert(sh.numel == len(arr));
        if (sh.numel > 0)
            setSliceImpl(_impl, sh.start, sh.stop, sh.step, getDataPtr<double>(arr), sh.numel);
    }

    friend std::ostream& operator<<(std::ostream& ss, const CudaRNVectorImpl& v) {
        printData(v._impl, std::ostream_iterator<float>(ss, " "));
        return ss;
    }

    const size_t _size;
    device_vector_ptr _impl;
};

class CudaRNImpl {
  public:
    CudaRNImpl(size_t n)
        : _n(n) {
    }

    CudaRNVectorImpl zero() {
        return CudaRNVectorImpl(_n, 0.0f);
    }

    CudaRNVectorImpl empty() {
        return CudaRNVectorImpl(_n);
    }

    void linComb(float a, const CudaRNVectorImpl& v1, float b, CudaRNVectorImpl& v2) {
        linCombImpl(a, v1._impl, b, v2._impl);
    }

    float inner(const CudaRNVectorImpl& v1, const CudaRNVectorImpl& v2) {
        return innerProduct(v1._impl, v2._impl);
    }

  private:
    size_t _n;
};

void convolution(const CudaRNVectorImpl& source, const CudaRNVectorImpl& kernel, CudaRNVectorImpl& target) {
	convImpl(source._impl, kernel._impl, target._impl);
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(PyCuda) {
    import_array(); //Import numpy

    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	def("conv", convolution);

    //typedef ClassWrapper<VectorXd, id<size_t>> EigenVector1;
    class_<CudaRNImpl>("CudaRNImpl", "Documentation",
                       init<size_t>())
        .def("zero", &CudaRNImpl::zero)
        .def("empty", &CudaRNImpl::empty)
        .def("linComb", &CudaRNImpl::linComb)
        .def("inner", &CudaRNImpl::inner);

    class_<CudaRNVectorImpl>("CudaRNVectorImpl", "Documentation",
                             init<size_t>())
        .def(init<size_t, float>())
        .def(init<numeric::array>())
        .def("assign", &CudaRNVectorImpl::assign)
        .def(self_ns::str(self_ns::self))
        .def("__getitem__", &CudaRNVectorImpl::getItem)
        .def("__setitem__", &CudaRNVectorImpl::setItem)
        .def("getSlice", &CudaRNVectorImpl::getSlice)
        .def("setSlice", &CudaRNVectorImpl::setSlice);
}