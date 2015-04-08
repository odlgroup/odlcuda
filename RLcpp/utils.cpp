#pragma once

#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <RLcpp/numpy_eigen.h>
#include <RLcpp/wrap.h>
#include <RLcpp/utils/StandardPhantoms.h>
#include <RLcpp/utils/Phantom.h>

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
    import_array(); //Import numpy

    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    def("greet", &greet);

    def("phantom", &phantom, phantom_overloads());

    enum_<RLCpp::PhantomType>("PhantomType", "Enumeration of available phantoms")
		.value("sheppLogan", RLCpp::PhantomType::sheppLogan)
		.value("modifiedSheppLogan", RLCpp::PhantomType::modifiedSheppLogan)
		.value("twoEllipses", RLCpp::PhantomType::twoEllipses)
		.value("circle", RLCpp::PhantomType::circle);
}