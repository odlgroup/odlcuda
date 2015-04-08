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
typedef std::shared_ptr<thrust::device_vector<float>> device_vector_ptr;

//Externally (CUDA) compiled

//Create
extern device_vector_ptr makeThrustVector(size_t size);
extern device_vector_ptr makeThrustVector(size_t size, float value);

//Algebra
extern void linCombImpl(float a, const device_vector_ptr& v1, float b, device_vector_ptr& v2);
extern float innerImpl(const device_vector_ptr& v1, const device_vector_ptr& v2);
extern void multiplyImpl(const device_vector_ptr& v1, device_vector_ptr& v2);

//Transformations
extern void absImpl(const device_vector_ptr& source, device_vector_ptr& target);
extern void negImpl(const device_vector_ptr& source, device_vector_ptr& target);

//Reductions
extern float normSqImpl(const device_vector_ptr& v1);
extern float sumImpl(const device_vector_ptr& v);

//Copy methods
extern void copyHostToDevice(double* source, device_vector_ptr& target);
extern void copyDeviceToHost(const device_vector_ptr& source, double* target);
extern void printData(const device_vector_ptr& v1, std::ostream_iterator<float>& out);

//Getters and setters
extern float getItemImpl(const device_vector_ptr& v1, int index);
extern void setItemImpl(device_vector_ptr& v1, int index, float value);
extern void getSliceImpl(const device_vector_ptr& v1, int begin, int end, int step, double* target);
extern void setSliceImpl(const device_vector_ptr& v1, int begin, int end, int step, double* source, int num);

//Functions
extern void convImpl(const device_vector_ptr& source, const device_vector_ptr& kernel, device_vector_ptr& target);

extern void forwardDifferenceImpl(const device_vector_ptr& source, device_vector_ptr& target);
extern void forwardDifferenceAdjointImpl(const device_vector_ptr& source, device_vector_ptr& target);

extern void forwardDifference2DImpl(const device_vector_ptr& source, device_vector_ptr& dx, device_vector_ptr& dy, int cols, int rows);
extern void forwardDifference2DAdjointImpl(const device_vector_ptr& dx, const device_vector_ptr& dy, device_vector_ptr& target, int cols, int rows);

extern void maxVectorVectorImpl(const device_vector_ptr& v1, device_vector_ptr& v2);
extern void maxVectorScalarImpl(const device_vector_ptr& v1, float scalar, device_vector_ptr& target);
extern void addScalarImpl(const device_vector_ptr& v1, float scalar, device_vector_ptr& target);
extern void signImpl(const device_vector_ptr& v1, device_vector_ptr& target);

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

    operator device_vector_ptr&() {
        return _impl;
    }

    operator device_vector_ptr const&() const {
        return _impl;
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
        assert(v1._size == v2._size);

        linCombImpl(a, v1._impl, b, v2._impl);
    }

    float inner(const CudaRNVectorImpl& v1, const CudaRNVectorImpl& v2) {
        return innerImpl(v1._impl, v2._impl);
    }

    float normSq(const CudaRNVectorImpl& v) const {
        return normSqImpl(v._impl);
    }

    void multiply(const CudaRNVectorImpl& v1, CudaRNVectorImpl& v2) {
        multiplyImpl(v1._impl, v2._impl);
    }

  private:
    size_t _n;
};

void convolution(const CudaRNVectorImpl& source, const CudaRNVectorImpl& kernel, CudaRNVectorImpl& target) {
    convImpl(source._impl, kernel._impl, target._impl);
}

void forwardDifference(const CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    forwardDifferenceImpl(source._impl, target._impl);
}

void forwardDifferenceAdjoint(const CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    forwardDifferenceAdjointImpl(source._impl, target._impl);
}

void forwardDifference2D(const CudaRNVectorImpl& source, CudaRNVectorImpl& dx, CudaRNVectorImpl& dy, int cols, int rows) {
	forwardDifference2DImpl(source._impl, dx._impl, dy._impl, cols, rows);
}

void forwardDifference2DAdjoint(const CudaRNVectorImpl& dx, const CudaRNVectorImpl& dy, CudaRNVectorImpl& target, int cols, int rows) {
	forwardDifference2DAdjointImpl(dx._impl, dy._impl, target._impl, cols, rows);
}

void maxVectorVector(const CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    maxVectorVectorImpl(source._impl, target._impl);
}

void maxVectorScalar(CudaRNVectorImpl& source, float scalar, CudaRNVectorImpl& target) {
    maxVectorScalarImpl(source._impl, scalar, target._impl);
}

void addScalar(const CudaRNVectorImpl& source, float scalar, CudaRNVectorImpl& target) {
    addScalarImpl(source._impl, scalar, target._impl);
}

void sign(const CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    signImpl(source._impl, target._impl);
}

void absVector(CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    absImpl(source._impl, target._impl);
}

float sumVector(const CudaRNVectorImpl& source) {
    return sumImpl(source._impl);
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(PyCuda) {
    import_array(); //Import numpy

    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	//CudaRNVectorImpl_converter().from_python(); todo

    def("conv", convolution);
    def("forwardDiff", forwardDifference);
    def("forwardDiffAdj", forwardDifferenceAdjoint);
	def("forwardDiff2D", forwardDifference2D);
	def("forwardDiff2DAdj", forwardDifference2DAdjoint);
    def("maxVectorVector", maxVectorVector);
    def("maxVectorScalar", maxVectorScalar);
    def("addScalar", addScalar);
    def("sign", sign);
    def("abs", absVector);
    def("sum", sumVector);

    //typedef ClassWrapper<VectorXd, id<size_t>> EigenVector1;
    class_<CudaRNImpl>("CudaRNImpl", "Documentation",
                       init<size_t>())
        .def("zero", &CudaRNImpl::zero)
        .def("empty", &CudaRNImpl::empty)
        .def("linComb", &CudaRNImpl::linComb)
        .def("inner", &CudaRNImpl::inner)
        .def("multiply", &CudaRNImpl::multiply)
        .def("normSq", &CudaRNImpl::normSq);

    class_<CudaRNVectorImpl>("CudaRNVectorImpl", "Documentation",
                             init<size_t>())
        .def(init<size_t, float>())
        .def(init<numeric::array>())
        .def(self_ns::str(self_ns::self))
        .def("__getitem__", &CudaRNVectorImpl::getItem)
        .def("__setitem__", &CudaRNVectorImpl::setItem)
        .def("getSlice", &CudaRNVectorImpl::getSlice)
        .def("setSlice", &CudaRNVectorImpl::setSlice);
}