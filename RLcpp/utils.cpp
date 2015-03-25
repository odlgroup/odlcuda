#pragma once

#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <RLcpp/numpy_eigen.h>
#include <RLcpp/wrap.h>

using namespace boost::python;
using namespace Eigen;

class EigenVector {
  public:
    EigenVector(size_t size)
        : impl(size) {
    }

    EigenVector(VectorXd&& other)
        : impl(std::forward<VectorXd>(other)) {
    }

    EigenVector(const numeric::array& data)
        : impl(copyInput<VectorXd>(data)) {
    }

    EigenVector(const list& data)
        : impl(len(data)) {
        for (auto i = 0; i < len(data); ++i) {
            impl[i] = extract<double>(data[i]);
        }
    }

    friend EigenVector operator+(const EigenVector& v1, const EigenVector& v2) {
        return EigenVector(v1.impl + v2.impl);
    }

    friend EigenVector operator*(const double& a, const EigenVector& v) {
        return EigenVector(a * v.impl);
    }

    friend EigenVector operator*(const EigenVector& v, const double& a) {
        return EigenVector(a * v.impl);
    }

    friend std::ostream& operator<<(std::ostream& ss, const EigenVector& v) {
        return ss << v.impl;
    }

  private:
    VectorXd impl;
};

char const* greet() {
    return "hello, world";
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(PyUtils) {
    import_array(); //Import numpy

    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("greet", &greet);

    //typedef ClassWrapper<VectorXd, id<size_t>> EigenVector1;
    class_<EigenVector>("EigenVector", "Documentation",
                        init<size_t>())
        .def(init<numeric::array>())
        .def(init<list>())
        .def(self + self)
        .def(self * double())
        .def(self_ns::str(self_ns::self));
}