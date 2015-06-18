// Disable deprecated API
#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//Thrust bug...
#define DEBUG 1
#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <iostream>
#define _DEBUG 1
#include <thrust/device_vector.h>
#include <numpy/arrayobject.h>

#include <RLcpp/numpy_utils.h>
#include <RLcpp/DeviceVector.h>

using namespace boost::python;

//Externally (CUDA) compiled

//Create
template <typename T>
extern DeviceVectorPtr<T> makeThrustVector(size_t size);

template <typename T>
extern DeviceVectorPtr<T> makeThrustVector(size_t size, T value);

template <typename T>
extern DeviceVectorPtr<T> makeWrapperVector(T * data, size_t size);

//Get ptr
template <typename T>
extern T* getRawPtr(DeviceVector<T>& ptr);
template <typename T>
extern T const* getRawPtr(const DeviceVector<T>& ptr);

//Algebra
extern void linCombImpl(DeviceVector<float>& z, float a, const DeviceVector<float>& x, float b, const DeviceVector<float>& y);
extern float innerImpl(const DeviceVector<float>& v1, const DeviceVector<float>& v2);
extern void multiplyImpl(const DeviceVector<float>& v1, DeviceVector<float>& v2);

//Transformations
extern void absImpl(const DeviceVector<float>& source, DeviceVector<float>& target);
extern void negImpl(const DeviceVector<float>& source, DeviceVector<float>& target);

//Reductions
extern float normImpl(const DeviceVector<float>& v1);
extern float sumImpl(const DeviceVector<float>& v);

//Copy methods
extern void printData(const DeviceVector<float>& v1, std::ostream_iterator<float>& out, int numel);

//Getters and setters
extern float getItemImpl(const DeviceVector<float>& v1, int index);
extern void setItemImpl(DeviceVector<float>& v1, int index, float value);
extern void getSliceImpl(const DeviceVector<float>& v1, int begin, int end, int step, double* target);
extern void setSliceImpl(DeviceVector<float>& v1, int begin, int end, int step, double* source, int num);

//Functions
extern void convImpl(const DeviceVector<float>& source, const DeviceVector<float>& kernel, DeviceVector<float>& target);

extern void forwardDifferenceImpl(const DeviceVector<float>& source, DeviceVector<float>& target);
extern void forwardDifferenceAdjointImpl(const DeviceVector<float>& source, DeviceVector<float>& target);

extern void forwardDifference2DImpl(const DeviceVector<float>& source, DeviceVector<float>& dx, DeviceVector<float>& dy, int cols, int rows);
extern void forwardDifference2DAdjointImpl(const DeviceVector<float>& dx, const DeviceVector<float>& dy, DeviceVector<float>& target, int cols, int rows);

extern void maxVectorVectorImpl(const DeviceVector<float>& v1, const DeviceVector<float>& v2, DeviceVector<float>& target);
extern void maxVectorScalarImpl(const DeviceVector<float>& v1, float scalar, DeviceVector<float>& target);
extern void divideVectorVectorImpl(const DeviceVector<float>& dividend, const DeviceVector<float>& divisor, DeviceVector<float>& quotient);

extern void addScalarImpl(const DeviceVector<float>& v1, float scalar, DeviceVector<float>& target);
extern void signImpl(const DeviceVector<float>& v1, DeviceVector<float>& target);
extern void sqrtImpl(const DeviceVector<float>& v1, DeviceVector<float>& target);

struct sliceHelper {
    sliceHelper(const slice& index, int n) : arraySize(n) {
        extract<int> stepIn(index.step());
        if (stepIn.check())
            step = stepIn();
        else
            step = 1;

        if (step == 0)
            throw std::invalid_argument("step = 0 is not valid");

        extract<int> startIn(index.start());
        if (startIn.check()) {
            if (step > 0) {
                start = startIn();
                if (start < 0) start += n;
            } else {
                start = startIn() + 1;
                if (start <= 0) start += n;
            }
        } else if (step > 0)
            start = 0;
        else
            start = n;

        extract<int> stopIn(index.stop());
        if (stopIn.check()) {
            if (step > 0) {
                stop = stopIn();
                if (stop < 0) stop += n;
            } else {
                stop = stopIn() + 1;
                if (stop <= 0) stop += n;
            }
        } else if (step > 0)
            stop = n;
        else
            stop = 0;

        if (start == stop)
            numel = 0;
        else if (step > 0)
            numel = std::max(0, 1 + (stop - start - 1) / step);
        else
            numel = std::max(0, 1 + (start - stop - 1) / std::abs(step));

        if (start < 0 || stop > arraySize)
            throw std::out_of_range("Slice index out of range");
    }

    friend std::ostream& operator<<(std::ostream& ss, const sliceHelper& sh) {
        return ss << sh.start << " " << sh.stop << " " << sh.step << " " << sh.numel;
    }
    int start, stop, step, numel, arraySize;
};

class CudaRNVectorImpl {
  public:
    CudaRNVectorImpl(size_t size)
        : _size(size),
          _impl(makeThrustVector<float>(size)) {
    }

    CudaRNVectorImpl(size_t size, float value)
        : _size(size),
          _impl(makeThrustVector<float>(size, value)) {
    }

	CudaRNVectorImpl(size_t size, DeviceVectorPtr<float> impl)
        : _size(size),
          _impl(impl) {
    }

    void validateIndex(ptrdiff_t index) const {
        if (index < 0 || index >= _size)
            throw std::out_of_range("index out of range");
    }

    float getItem(ptrdiff_t index) const {
        if (index < 0) index += _size; //Handle negative indexes like python
        validateIndex(index);
        return getItemImpl(*this, index);
    }

    void setItem(ptrdiff_t index, float value) {
        if (index < 0) index += _size; //Handle negative indexes like python
        validateIndex(index);
        setItemImpl(*this, index, value);
    }

    numeric::array getSlice(const slice index) const {
        sliceHelper sh(index, _size);

        if (sh.numel > 0) {
            numeric::array arr = makeArray<double>(sh.numel);
            getSliceImpl(*this, sh.start, sh.stop, sh.step, getDataPtr<double>(arr));
            return arr;
        } else {
            return makeArray<double>(0);
        }
    }

    void setSlice(const slice index, const numeric::array& arr) {
        sliceHelper sh(index, _size);

        if (sh.numel != len(arr))
            throw std::out_of_range("Size of array does not match slice");

        if (sh.numel > 0) {
            setSliceImpl(*this, sh.start, sh.stop, sh.step, getDataPtr<double>(arr), sh.numel);
        }
    }

    friend std::ostream& operator<<(std::ostream& ss, const CudaRNVectorImpl& v) {
        ss << "CudaRNVectorImpl: ";
        auto outputIter = std::ostream_iterator<float>(ss, " ");
        printData(v, outputIter, std::min<int>(100, v._size));
        return ss;
    }

    operator DeviceVector<float>&() {
        return *_impl;
    }

    operator DeviceVector<float> const&() const {
        return *_impl;
    }

    object convertFloat(float* ptr) {
        object ctypes = import("ctypes");
        long_ addr(reinterpret_cast<uintptr_t>(ptr));

        object POINTER = ctypes.attr("POINTER");
        object c_float = ctypes.attr("c_float");
        object pointerType = POINTER(c_float);

        return ctypes.attr("cast")(addr, pointerType);
    }

    uintptr_t dataPtr() {
        return reinterpret_cast<uintptr_t>(getRawPtr(*_impl));
    }

    const size_t _size;
	DeviceVectorPtr<float> _impl;
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

    void linComb(CudaRNVectorImpl& z, float a, const CudaRNVectorImpl& x, float b, CudaRNVectorImpl& y) {
        assert(z._size == x._size);
        assert(z._size == y._size);

        linCombImpl(z, a, x, b, y);
    }

    float inner(const CudaRNVectorImpl& v1, const CudaRNVectorImpl& v2) {
        return innerImpl(v1, v2);
    }

    float norm(const CudaRNVectorImpl& v) const {
        return normImpl(v);
    }

    void multiply(const CudaRNVectorImpl& v1, CudaRNVectorImpl& v2) {
        multiplyImpl(v1, v2);
    }

  private:
    size_t _n;
};

void convolution(const CudaRNVectorImpl& source, const CudaRNVectorImpl& kernel, CudaRNVectorImpl& target) {
    convImpl(source, kernel, target);
}

void forwardDifference(const CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    forwardDifferenceImpl(source, target);
}

void forwardDifferenceAdjoint(const CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    forwardDifferenceAdjointImpl(source, target);
}

void forwardDifference2D(const CudaRNVectorImpl& source, CudaRNVectorImpl& dx, CudaRNVectorImpl& dy, int cols, int rows) {
    forwardDifference2DImpl(source, dx, dy, cols, rows);
}

void forwardDifference2DAdjoint(const CudaRNVectorImpl& dx, const CudaRNVectorImpl& dy, CudaRNVectorImpl& target, int cols, int rows) {
    forwardDifference2DAdjointImpl(dx, dy, target, cols, rows);
}

void maxVectorVector(const CudaRNVectorImpl& v1, const CudaRNVectorImpl& v2, CudaRNVectorImpl& target) {
    maxVectorVectorImpl(v1, v2, target);
}

void maxVectorScalar(CudaRNVectorImpl& source, float scalar, CudaRNVectorImpl& target) {
    maxVectorScalarImpl(source, scalar, target);
}

void divideVectorVector(const CudaRNVectorImpl& dividend, const CudaRNVectorImpl& divisor, CudaRNVectorImpl& quotient) {
    divideVectorVectorImpl(dividend, divisor, quotient);
}

void addScalar(const CudaRNVectorImpl& source, float scalar, CudaRNVectorImpl& target) {
    addScalarImpl(source, scalar, target);
}

void signVector(const CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    signImpl(source, target);
}

void sqrtVector(const CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    sqrtImpl(source, target);
}

void absVector(CudaRNVectorImpl& source, CudaRNVectorImpl& target) {
    absImpl(source, target);
}

float sumVector(const CudaRNVectorImpl& source) {
    return sumImpl(source);
}

CudaRNVectorImpl vectorFromPointer(uintptr_t ptr, size_t size) {
    return CudaRNVectorImpl{size, makeWrapperVector((float*)ptr, size)};
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(PyCuda) {
    auto result = _import_array(); //Import numpy
    if (result < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return;
    }
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("conv", convolution);
    def("forwardDiff", forwardDifference);
    def("forwardDiffAdj", forwardDifferenceAdjoint);
    def("forwardDiff2D", forwardDifference2D);
    def("forwardDiff2DAdj", forwardDifference2DAdjoint);
    def("maxVectorVector", maxVectorVector);
    def("maxVectorScalar", maxVectorScalar);
    def("divideVectorVector", divideVectorVector);
    def("addScalar", addScalar);
    def("sign", signVector);
    def("sqrt", sqrtVector);
    def("abs", absVector);
    def("sum", sumVector);

    def("vectorFromPointer", vectorFromPointer);

    class_<CudaRNImpl>("CudaRNImpl", "Documentation",
                       init<size_t>())
        .def("zero", &CudaRNImpl::zero)
        .def("empty", &CudaRNImpl::empty)
        .def("linComb", &CudaRNImpl::linComb)
        .def("inner", &CudaRNImpl::inner)
        .def("multiply", &CudaRNImpl::multiply)
        .def("norm", &CudaRNImpl::norm);

    class_<CudaRNVectorImpl>("CudaRNVectorImpl", "Documentation",
                             init<size_t>())
        .def(init<size_t, float>())
        .def(self_ns::str(self_ns::self))
        .def("__getitem__", &CudaRNVectorImpl::getItem)
        .def("__setitem__", &CudaRNVectorImpl::setItem)
        .def("getSlice", &CudaRNVectorImpl::getSlice)
        .def("setSlice", &CudaRNVectorImpl::setSlice)
        .def("dataPtr", &CudaRNVectorImpl::dataPtr);
}
