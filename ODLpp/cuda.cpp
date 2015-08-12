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

#include <LCRUtils/python/numpy_utils.h>
#include <ODLpp/DeviceVector.h>

using namespace boost::python;

//Externally (CUDA) compiled

template <typename T>
struct CudaRNVectorImplMethods {
    // Creation
    static DeviceVectorPtr<T> makeThrustVector(size_t size);
    static DeviceVectorPtr<T> makeThrustVector(size_t size, T value);
    static DeviceVectorPtr<T> makeWrapperVector(T* data, size_t size);

    //Get ptr
    static T* getRawPtr(DeviceVector<T>& ptr);
    static const T* getRawPtr(const DeviceVector<T>& ptr);

    //Getters and setters
    static T getItemImpl(const DeviceVector<T>& v1, int index);
    static void setItemImpl(DeviceVector<T>& v1, int index, T value);
    static void getSliceImpl(const DeviceVector<T>& v1, int begin, int end, int step, T* target);
		static void setSliceImpl(DeviceVector<T>& v1, int begin, int end, int step, const T* source, int num);

    //Algebra
    static void linCombImpl(DeviceVector<T>& z, T a, const DeviceVector<T>& x, T b, const DeviceVector<T>& y);
    static double normImpl(const DeviceVector<T>& v1);
    static double innerImpl(const DeviceVector<float>& v1, const DeviceVector<float>& v2);
    static void multiplyImpl(const DeviceVector<float>& v1, DeviceVector<float>& v2);

    //Copy methods
    static void printData(const DeviceVector<T>& v1, std::ostream_iterator<T>& out, int numel);
};

//Transformations
extern void absImpl(const DeviceVector<float>& source, DeviceVector<float>& target);
extern void negImpl(const DeviceVector<float>& source, DeviceVector<float>& target);

//Reductions
extern float sumImpl(const DeviceVector<float>& v);

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
    sliceHelper(const slice& index, ptrdiff_t n) : arraySize(n) {
        extract<ptrdiff_t> stepIn(index.step());
        if (stepIn.check())
            step = stepIn();
        else
            step = 1;

        if (step == 0)
            throw std::invalid_argument("step = 0 is not valid");

        extract<ptrdiff_t> startIn(index.start());
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

        extract<ptrdiff_t> stopIn(index.stop());
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
            numel = std::max<ptrdiff_t>(0, 1 + (stop - start - 1) / step);
        else
            numel = std::max<ptrdiff_t>(0, 1 + (start - stop - 1) / std::abs(step));

        if (start < 0 || stop > arraySize)
            throw std::out_of_range("Slice index out of range");
    }

    friend std::ostream& operator<<(std::ostream& ss, const sliceHelper& sh) {
        return ss << sh.start << " " << sh.stop << " " << sh.step << " " << sh.numel;
    }
    ptrdiff_t start, stop, step, numel, arraySize;
};

template <typename T>
class CudaVectorImpl {
  public:
    CudaVectorImpl(size_t size)
        : _size(size),
          _impl(CudaRNVectorImplMethods<T>::makeThrustVector(size)) {
    }

    CudaVectorImpl(size_t size, T value)
        : _size(size),
          _impl(CudaRNVectorImplMethods<T>::makeThrustVector(size, value)) {
    }

    CudaVectorImpl(size_t size, DeviceVectorPtr<T> impl)
        : _size(size),
          _impl(impl) {
    }

    static CudaVectorImpl<T> fromPointer(uintptr_t ptr, size_t size) {
        return CudaVectorImpl<T>{size, CudaRNVectorImplMethods<T>::makeWrapperVector((T*)ptr, size)};
    }

    void validateIndex(ptrdiff_t index) const {
        if (index < 0 || index >= _size)
            throw std::out_of_range("index out of range");
    }

    T getItem(ptrdiff_t index) const {
        if (index < 0) index += _size; //Handle negative indexes like python
        validateIndex(index);
        return CudaRNVectorImplMethods<T>::getItemImpl(*_impl, index);
    }

    void setItem(ptrdiff_t index, T value) {
        if (index < 0) index += _size; //Handle negative indexes like python
        validateIndex(index);
        CudaRNVectorImplMethods<T>::setItemImpl(*_impl, index, value);
    }

    numeric::array getSlice(const slice index) const {
        sliceHelper sh(index, _size);

        if (sh.numel > 0) {
            numeric::array arr = makeArray<T>(sh.numel);
            CudaRNVectorImplMethods<T>::getSliceImpl(*_impl, sh.start, sh.stop, sh.step, getDataPtr<T>(arr));
            return arr;
        } else {
            return makeArray<T>(0);
        }
    }

    void setSlice(const slice index, const numeric::array& arr) {
        sliceHelper sh(index, _size);

        if (sh.numel != len(arr))
            throw std::out_of_range("Size of array does not match slice");

        if (sh.numel > 0) {
            CudaRNVectorImplMethods<T>::setSliceImpl(*_impl, sh.start, sh.stop, sh.step, getDataPtr<T>(arr), sh.numel);
        }
    }

    void linComb(T a, const CudaVectorImpl<T>& x, T b, const CudaVectorImpl<T>& y) {
        CudaRNVectorImplMethods<T>::linCombImpl(*this, a, x, b, y);
    }

    double inner(const CudaVectorImpl<T>& v2) const {
        return CudaRNVectorImplMethods<T>::innerImpl(*this, v2);
    }

    double norm() const {
        return CudaRNVectorImplMethods<T>::normImpl(*this);
    }

    void multiply(const CudaVectorImpl<T>& v1) {
        CudaRNVectorImplMethods<T>::multiplyImpl(v1, *this);
    }

    friend std::ostream& operator<<(std::ostream& ss, const CudaVectorImpl& v) {
        ss << "CudaVectorImpl" << typeid(T).name() << ">: ";
        auto outputIter = std::ostream_iterator<T>(ss, " ");
        CudaRNVectorImplMethods<T>::printData(*v._impl, outputIter, std::min<int>(100, v._size));
        return ss;
    }

    operator DeviceVector<T>&() {
        return *_impl;
    }

    operator DeviceVector<T> const&() const {
        return *_impl;
    }

    uintptr_t dataPtr() {
        return reinterpret_cast<uintptr_t>(CudaRNVectorImplMethods<T>::getRawPtr(*_impl));
    }

    const size_t _size;
    DeviceVectorPtr<T> _impl;
};

// Functions
void convolution(const CudaVectorImpl<float>& source, const CudaVectorImpl<float>& kernel, CudaVectorImpl<float>& target) {
    convImpl(source, kernel, target);
}

void forwardDifference(const CudaVectorImpl<float>& source, CudaVectorImpl<float>& target) {
    forwardDifferenceImpl(source, target);
}

void forwardDifferenceAdjoint(const CudaVectorImpl<float>& source, CudaVectorImpl<float>& target) {
    forwardDifferenceAdjointImpl(source, target);
}

void forwardDifference2D(const CudaVectorImpl<float>& source, CudaVectorImpl<float>& dx, CudaVectorImpl<float>& dy, int cols, int rows) {
    forwardDifference2DImpl(source, dx, dy, cols, rows);
}

void forwardDifference2DAdjoint(const CudaVectorImpl<float>& dx, const CudaVectorImpl<float>& dy, CudaVectorImpl<float>& target, int cols, int rows) {
    forwardDifference2DAdjointImpl(dx, dy, target, cols, rows);
}

void maxVectorVector(const CudaVectorImpl<float>& v1, const CudaVectorImpl<float>& v2, CudaVectorImpl<float>& target) {
    maxVectorVectorImpl(v1, v2, target);
}

void maxVectorScalar(CudaVectorImpl<float>& source, float scalar, CudaVectorImpl<float>& target) {
    maxVectorScalarImpl(source, scalar, target);
}

void divideVectorVector(const CudaVectorImpl<float>& dividend, const CudaVectorImpl<float>& divisor, CudaVectorImpl<float>& quotient) {
    divideVectorVectorImpl(dividend, divisor, quotient);
}

void addScalar(const CudaVectorImpl<float>& source, float scalar, CudaVectorImpl<float>& target) {
    addScalarImpl(source, scalar, target);
}

void signVector(const CudaVectorImpl<float>& source, CudaVectorImpl<float>& target) {
    signImpl(source, target);
}

void sqrtVector(const CudaVectorImpl<float>& source, CudaVectorImpl<float>& target) {
    sqrtImpl(source, target);
}

void absVector(CudaVectorImpl<float>& source, CudaVectorImpl<float>& target) {
    absImpl(source, target);
}

float sumVector(const CudaVectorImpl<float>& source) {
    return sumImpl(source);
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(odlpp_cuda) {
    auto result = _import_array(); // Import numpy
    if (result < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return;
    }
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("conv", convolution);
    def("forward_diff", forwardDifference);
    def("forward_diff_adj", forwardDifferenceAdjoint);
    def("forward_diff_2d", forwardDifference2D);
    def("forward_diff_2d_adj", forwardDifference2DAdjoint);
    def("max_vector_vector", maxVectorVector);
    def("max_vector_scalar", maxVectorScalar);
    def("divide_vector_vector", divideVectorVector);
    def("add_scalar", addScalar);
    def("sign", signVector);
    def("sqrt", sqrtVector);
    def("abs", absVector);
    def("sum", sumVector);

    // CudaRn
    class_<CudaVectorImpl<float>>("CudaVectorFloat", "Documentation",
                                  init<size_t>())
        .def(init<size_t, float>())
        .def("from_pointer", &CudaVectorImpl<float>::fromPointer)
        .staticmethod("from_pointer")
        .def(self_ns::str(self_ns::self))
        .def("__getitem__", &CudaVectorImpl<float>::getItem)
        .def("__setitem__", &CudaVectorImpl<float>::setItem)
        .def("getslice", &CudaVectorImpl<float>::getSlice)
        .def("setslice", &CudaVectorImpl<float>::setSlice)
        .def("data_ptr", &CudaVectorImpl<float>::dataPtr)
				.def("linComb", &CudaVectorImpl<float>::linComb)
        .def("inner", &CudaVectorImpl<float>::inner)
        .def("norm", &CudaVectorImpl<float>::norm)
        .def("multiply", &CudaVectorImpl<float>::multiply);

    // CudaEn with uchar
    class_<CudaVectorImpl<unsigned char>>("CudaVectorUchar", "Documentation",
                                          init<size_t>())
        .def(init<size_t, unsigned char>())
        .def("from_pointer", &CudaVectorImpl<unsigned char>::fromPointer)
        .staticmethod("from_pointer")
        .def(self_ns::str(self_ns::self))
        .def("__getitem__", &CudaVectorImpl<unsigned char>::getItem)
        .def("__setitem__", &CudaVectorImpl<unsigned char>::setItem)
        .def("getslice", &CudaVectorImpl<unsigned char>::getSlice)
        .def("setslice", &CudaVectorImpl<unsigned char>::setSlice)
        .def("data_ptr", &CudaVectorImpl<unsigned char>::dataPtr);
}
