#pragma once

#include <Eigen/Dense>

namespace ODLpp {
/**
 * Definition of an ellipse in two dimensions, also contains an greyLevel value.
 * Intended to be used to describe areas in phantoms.
 */
struct Ellipse {
    Ellipse(Eigen::Vector2d _center, double _majorAxis, double _minorAxis, double _theta, double _greyLevel)
        : center(_center),
          majorAxis(_majorAxis),
          minorAxis(_minorAxis),
          theta(_theta),
          greyLevel(_greyLevel) {
    }

    const Eigen::Vector2d center; /// Position of the center of the ellipse
    const double majorAxis;       /// Axis length at theta = 0
    const double minorAxis;       /// Axis length at theta = pi/2
    const double theta;           ///Rotation of ellipse (in radians)
    const double greyLevel;       /// Grey level value inside ellipse
};
}