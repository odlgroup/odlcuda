#include <ODLpp/odlutils/StandardPhantoms.h>
#include <ODLpp/odlutils/Phantom.h>
#include <ODLpp/odlutils/EigenUtils.h>
#include <stdexcept> // std::invalid_argument
#include <vector>
#include <iostream>
#include <cmath>

using namespace Eigen;

namespace ODLpp {
namespace {
/**
 * The Shepp-Logan ellipse parameters, as defined in:
 * http://stat.wharton.upenn.edu/~shepp/publications/33.pdf
 */
std::vector<Ellipse> sheppLoganEllipses() {
    double degToRad = M_PI / 180.0;

    std::vector<Ellipse> ellipses;
    ellipses.emplace_back(Vector2d(0.0, 0.0), 0.69, 0.92, 0.0, 2.0);
    ellipses.emplace_back(Vector2d(0.0, -0.0184), 0.6624, 0.874, 0.0, -0.98);
    ellipses.emplace_back(Vector2d(0.22, 0.0), 0.11, 0.31, -18 * degToRad, -0.02);
    ellipses.emplace_back(Vector2d(-0.22, 0.0), 0.16, 0.41, 18 * degToRad, -0.02);
    ellipses.emplace_back(Vector2d(0.0, 0.35), 0.21, 0.25, 0.0, 0.01);
	ellipses.emplace_back(Vector2d(0.0, 0.1), 0.046, 0.046, 0.0, 0.01);
	ellipses.emplace_back(Vector2d(0.0, -0.1), 0.046, 0.046, 0.0, 0.01);
    ellipses.emplace_back(Vector2d(-0.08, -0.605), 0.046, 0.023, 0.0, 0.01);
    ellipses.emplace_back(Vector2d(0.0, -0.605), 0.023, 0.023, 0.0, 0.01);
    ellipses.emplace_back(Vector2d(0.06, -0.605), 0.023, 0.046, 0.0, 0.01);
    return ellipses;
}

/// Modified Shepp-Logan
std::vector<Ellipse> modifiedSheppLoganEllipses() {
    double degToRad = M_PI / 180.0;

    std::vector<Ellipse> ellipses;
    ellipses.emplace_back(Vector2d(0.0, 0.0), 0.69, 0.92, 0.0, 1.0);
    ellipses.emplace_back(Vector2d(0.0, -0.0184), 0.6624, 0.874, 0.0, -0.8);
    ellipses.emplace_back(Vector2d(0.22, 0.0), 0.11, 0.31, -18 * degToRad, -0.2);
    ellipses.emplace_back(Vector2d(-0.22, 0.0), 0.16, 0.41, 18 * degToRad, -0.2);
    ellipses.emplace_back(Vector2d(0.0, 0.35), 0.21, 0.25, 0.0, 0.1);
    ellipses.emplace_back(Vector2d(0.0, 0.1), 0.046, 0.046, 0.0, 0.1);
	ellipses.emplace_back(Vector2d(0.0, -0.1), 0.046, 0.046, 0.0, 0.1);
    ellipses.emplace_back(Vector2d(-0.08, -0.605), 0.046, 0.023, 0.0, 0.1);
    ellipses.emplace_back(Vector2d(0.0, -0.605), 0.023, 0.023, 0.0, 0.1);
    ellipses.emplace_back(Vector2d(0.06, -0.605), 0.023, 0.046, 0.0, 0.1);
    return ellipses;
}

/// two ellipses
std::vector<Ellipse> twoEllipsesEllipses() {
    double degToRad = M_PI / 180.0;

    std::vector<Ellipse> ellipses;
    ellipses.emplace_back(Vector2d(0.0, 0.0), 0.7, 0.8, 0.0, 1.0);
    ellipses.emplace_back(Vector2d(0.0, -0.2), 0.5, 0.4, 0.0, -0.5);
    return ellipses;
}

/// two ellipses
std::vector<Ellipse> circle() {
    double degToRad = M_PI / 180.0;

    std::vector<Ellipse> ellipses;
    ellipses.emplace_back(Vector2d(0.0, 0.0), 0.8, 0.8, 0.0, 1.0);
    return ellipses;
}
}

std::vector<Ellipse> getPhantomParameters(PhantomType type) {
    switch (type) {
    case PhantomType::sheppLogan:
        return sheppLoganEllipses();
    case PhantomType::modifiedSheppLogan:
        return modifiedSheppLoganEllipses();
    case PhantomType::twoEllipses:
        return twoEllipsesEllipses();
    case PhantomType::circle:
        return circle();
    default:
        throw std::invalid_argument("Invalid type name");
    }
}
}
