// Disable deprecated API
#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include <odl_cpp_utils/python/numpy_eigen.h>

#include <ODLpp/odlutils/StandardPhantoms.h>
#include <ODLpp/odlutils/Phantom.h>

using namespace boost::python;
using namespace Eigen;

numeric::array phantom(const boost::python::object& size,
                       ODLpp::PhantomType type = ODLpp::PhantomType::modifiedSheppLogan,
                       double edgeWidth = 0.0) {
    return copyOutput(phantom(copyInput<Vector2i>(size),
                              type,
                              edgeWidth));
}

BOOST_PYTHON_FUNCTION_OVERLOADS(phantom_overloads, phantom, 1, 3)

char const* greet() {
    return "Hello, world!";
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(odlpp_utils) {
    auto result = _import_array(); //Import numpy
    if (result < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return;
    }

    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("greet", &greet);

    def("phantom", &phantom, phantom_overloads());

    enum_<ODLpp::PhantomType>("PhantomType", "Enumeration of available phantoms")
		.value("shepp_logan", ODLpp::PhantomType::sheppLogan)
		.value("modified_shepp_logan", ODLpp::PhantomType::modifiedSheppLogan)
		.value("two_ellipses", ODLpp::PhantomType::twoEllipses)
		.value("circle", ODLpp::PhantomType::circle);
}
