#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <RLcpp/numpy_eigen.h>
#include <RLcpp/rlutils/StandardPhantoms.h>
#include <RLcpp/rlutils/Phantom.h>

using namespace boost::python;
using namespace Eigen;

numeric::array phantom(const boost::python::object& size,
                       RLCpp::PhantomType type = RLCpp::PhantomType::modifiedSheppLogan,
                       double edgeWidth = 0.0) {
    return copyOutput(phantom(copyInput<Vector2i>(size),
                              type,
                              edgeWidth));
}

BOOST_PYTHON_FUNCTION_OVERLOADS(phantom_overloads, phantom, 1, 3)

char const* greet() {
    return "hello, world";
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(PyUtils) {
    auto result = _import_array(); //Import numpy
    if (result < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return;
    }

    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("greet", &greet);

    def("phantom", &phantom, phantom_overloads());

    enum_<RLCpp::PhantomType>("PhantomType", "Enumeration of available phantoms")
		.value("sheppLogan", RLCpp::PhantomType::sheppLogan)
		.value("modifiedSheppLogan", RLCpp::PhantomType::modifiedSheppLogan)
		.value("twoEllipses", RLCpp::PhantomType::twoEllipses)
		.value("circle", RLCpp::PhantomType::circle);
}
