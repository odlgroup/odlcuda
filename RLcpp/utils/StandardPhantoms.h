#pragma once

#include <RLcpp/utils/utilities_export.h>
#include <RLcpp/utils/Ellipse.h>
#include <vector>

#include <Eigen/Dense>

namespace RLCpp {
/**
 * Enum of all phantom types supported by getPhantomParameters
 */
enum class PhantomType {
    sheppLogan,         /// the standard Shepp-Logan phantom
    modifiedSheppLogan, /// Shepp-Logan with improved contrast
    twoEllipses,        /// A very simple phantom with two structures
	circle				/// The most simple phantom (1 circle)
};

/**
 * Gets the ellipse parameters for the phantom given by @a type
 */
std::vector<Ellipse> UTILITIES_EXPORT getPhantomParameters(PhantomType type);
}
