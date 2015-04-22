#include <RLcpp/rlutils/Phantom.h>
#include <RLcpp/rlutils/EigenUtils.h>
#include <RLcpp/rlutils/Ellipse.h>

#include <vector>
#include <iostream>
#include <cmath>

using namespace Eigen;

namespace RLCpp {

namespace {
/**
 * Adds ellipse.greyLevel to all points inside the ellipse described by @a ellipse
 */
void addEllipse(const Ellipse& ellipse, const ArrayXXd& xPos, const ArrayXXd& yPos, double edgeWidth, ArrayXXd& data) {
    //Calculate the distance from ellipse center
    ArrayXXd xOffset = xPos - ellipse.center[0];
    ArrayXXd yOffset = yPos - ellipse.center[1];

    //Rotate the coordinate axis's according to rotation
    ArrayXXd xRot = xOffset * std::cos(ellipse.theta) + yOffset * std::sin(ellipse.theta);
    ArrayXXd yRot = -xOffset * std::sin(ellipse.theta) + yOffset * std::cos(ellipse.theta);

    //Calculate normalized distance
    assert(ellipse.minorAxis > 0.0);
    assert(ellipse.majorAxis > 0.0);
    ArrayXXd dNorm = xRot.square() / (ellipse.majorAxis * ellipse.majorAxis) + yRot.square() / (ellipse.minorAxis * ellipse.minorAxis);

    //Increment data whose normalized distance is >= 1
    if (edgeWidth == 0.0)
        data = (dNorm >= 1.0).select(data, data + ellipse.greyLevel);
    else
        data += ellipse.greyLevel / (1.0 + ((dNorm - 1.0) / edgeWidth).exp()); //Logistic curve smoothing
}
}

ArrayXXd phantom(Vector2i size, PhantomType type, double edgeWidth) {
    assert(size[0] > 0);
    assert(size[1] > 0);

    //Generate coordinates
    ArrayXXd xPos, yPos;
    meshgrid(VectorXd::LinSpaced(size[0], -1.0, 1.0),
             VectorXd::LinSpaced(size[1], -1.0, 1.0),
             xPos, yPos);

    //Initialize data
    ArrayXXd data = ArrayXXd::Zero(size[0], size[1]);

    //Add all ellipses
	for (auto&& ellipse : getPhantomParameters(type)) {
        addEllipse(ellipse, xPos, yPos, edgeWidth, data);
    }

    return data;
}
}
