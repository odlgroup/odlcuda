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

#include <odl_cpp_utils/python/numpy_utils.h>
#include <odlcuda/cuda/DeviceVector.h>
#include <odlcuda/cuda/UFunc.h>
#include <odlcuda/cuda/Reduction.h>
#include <odlcuda/cuda/TypeMacro.h>
#include <odlcuda/cuda/CudaVector.h>

using namespace boost::python;

// Externally (CUDA) compiled

// Functions
extern void convImpl(const DeviceVector<float>& source,
                     const DeviceVector<float>& kernel,
                     DeviceVector<float>& target);

extern void forwardDifferenceImpl(const DeviceVector<float>& source,
                                  DeviceVector<float>& target);
extern void forwardDifferenceAdjointImpl(const DeviceVector<float>& source,
                                         DeviceVector<float>& target);

extern void forwardDifference2DImpl(const DeviceVector<float>& source,
                                    DeviceVector<float>& dx,
                                    DeviceVector<float>& dy, int cols,
                                    int rows);
extern void forwardDifference2DAdjointImpl(const DeviceVector<float>& dx,
                                           const DeviceVector<float>& dy,
                                           DeviceVector<float>& target,
                                           int cols, int rows);

extern void maxVectorVectorImpl(const DeviceVector<float>& v1,
                                const DeviceVector<float>& v2,
                                DeviceVector<float>& target);
extern void maxVectorScalarImpl(const DeviceVector<float>& v1, float scalar,
                                DeviceVector<float>& target);
extern void divideVectorVectorImpl(const DeviceVector<float>& dividend,
                                   const DeviceVector<float>& divisor,
                                   DeviceVector<float>& quotient);

extern void addScalarImpl(const DeviceVector<float>& v1, float scalar,
                          DeviceVector<float>& out);

extern void gaussianBlurImpl(const DeviceVector<float>& image,
                             DeviceVector<float>& out,
                             DeviceVector<float>& temporary,
                             const int image_width, const int image_height,
                             const float sigma_x, const float sigma_y,
                             const int kernel_width, const int kernel_height);

// Functions
void convolution(const CudaVectorImpl<float>& source,
                 const CudaVectorImpl<float>& kernel,
                 CudaVectorImpl<float>& target) {
    convImpl(source, kernel, target);
}

void forwardDifference(const CudaVectorImpl<float>& source,
                       CudaVectorImpl<float>& target) {
    forwardDifferenceImpl(source, target);
}

void forwardDifferenceAdjoint(const CudaVectorImpl<float>& source,
                              CudaVectorImpl<float>& target) {
    forwardDifferenceAdjointImpl(source, target);
}

void forwardDifference2D(const CudaVectorImpl<float>& source,
                         CudaVectorImpl<float>& dx, CudaVectorImpl<float>& dy,
                         int cols, int rows) {
    forwardDifference2DImpl(source, dx, dy, cols, rows);
}

void forwardDifference2DAdjoint(const CudaVectorImpl<float>& dx,
                                const CudaVectorImpl<float>& dy,
                                CudaVectorImpl<float>& target, int cols,
                                int rows) {
    forwardDifference2DAdjointImpl(dx, dy, target, cols, rows);
}

void maxVectorVector(const CudaVectorImpl<float>& v1,
                     const CudaVectorImpl<float>& v2,
                     CudaVectorImpl<float>& target) {
    maxVectorVectorImpl(v1, v2, target);
}

void maxVectorScalar(CudaVectorImpl<float>& source, float scalar,
                     CudaVectorImpl<float>& target) {
    maxVectorScalarImpl(source, scalar, target);
}

void divideVectorVector(const CudaVectorImpl<float>& dividend,
                        const CudaVectorImpl<float>& divisor,
                        CudaVectorImpl<float>& quotient) {
    divideVectorVectorImpl(dividend, divisor, quotient);
}

void addScalar(const CudaVectorImpl<float>& source, float scalar,
               CudaVectorImpl<float>& target) {
    addScalarImpl(source, scalar, target);
}

extern void gaussianBlur(const CudaVectorImpl<float>& image,
                         CudaVectorImpl<float>& temporary,
                         CudaVectorImpl<float>& out, const int image_width,
                         const int image_height, const float sigma_x,
                         const float sigma_y, const int kernel_width,
                         const int kernel_height) {
    gaussianBlurImpl(image, temporary, out, image_width, image_height, sigma_x,
                     sigma_y, kernel_width, kernel_height);
}

template <typename T>
void instantiateCudaVector(const std::string& name) {
    using Float = typename CudaVectorImpl<T>::Float;
    auto cls =
        class_<CudaVectorImpl<T>>(name.c_str(), "Documentation", init<size_t>())
            .def(init<size_t, T>())
            .def("from_pointer", &fromPointer<T>)
            .staticmethod("from_pointer")
            .def("copy", &CudaVectorImpl<T>::copy)
            .def(self_ns::str(self_ns::self))
            .def("__repr__", &repr<T>)
            .def("data_ptr", &CudaVectorImpl<T>::dataPtr)
            .add_property("dtype", &dtype<T>)
            .add_property("shape", &shape<T>)
            .add_property("size", &CudaVectorImpl<T>::size)
            .add_property("itemsize", &itemsize<T>)
            .add_property("nbytes", &nbytes<T>)
            .def("__len__", &CudaVectorImpl<T>::size)
            .def("__eq__", &CudaVectorImpl<T>::allEqual)
            .def("__getitem__", &CudaVectorImpl<T>::getItem)
            .def("__setitem__", &CudaVectorImpl<T>::setItem)
            .def("copy_device_to_host", &copyDeviceToHost<T>)
            .def("get_to_host", &getSliceToHost<T>)
            .def("getslice", &getSliceView<T>)
            .def("setslice", &setSlice<T>)
            .def("lincomb", &CudaVectorImpl<T>::linComb)
            .def("fill", &CudaVectorImpl<T>::fill)
            .def("dist", &CudaVectorImpl<T>::dist)
            .def("dist_power", &CudaVectorImpl<T>::dist_power)
            .def("dist_weight", &CudaVectorImpl<T>::dist_weight)
            .def("norm", &CudaVectorImpl<T>::norm)
            .def("norm_power", &CudaVectorImpl<T>::norm_power)
            .def("norm_weight", &CudaVectorImpl<T>::norm_weight)
            .def("inner", &CudaVectorImpl<T>::inner)
            .def("inner_weight", &CudaVectorImpl<T>::inner_weight)
            .def("dist", &CudaVectorImpl<T>::dist)
            .def("norm", &CudaVectorImpl<T>::norm)
            .def("multiply", &CudaVectorImpl<T>::multiply)
            .def("divide", &CudaVectorImpl<T>::divide);

#define X(fun) cls.def(#fun, &ufunc_##fun<T, T>);
    ODL_CUDA_FOR_EACH_UFUNC
#undef X

#define X(fun) cls.def(#fun, &reduction_##fun<T>);
    ODL_CUDA_FOR_EACH_REDUCTION
#undef X
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(odlcuda_) {
    auto result = _import_array(); // Import numpy
    if (result != 0) {
        PyErr_Print();
        throw std::invalid_argument("numpy.core.multiarray failed to import");
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
    def("gaussianBlur", gaussianBlur);

// Instatiate according to numpy
#define X(type, name) instantiateCudaVector<type>(name);
    ODL_CUDA_FOR_EACH_TYPE
#undef X
}
