#pragma once

#include <TVLib/utils/utilities_export.h>
#include <TVLib/utils/StandardPhantoms.h>

#include <Eigen/Dense>

namespace TVLib {
/**
 * Constructs a phantom.
 * @param size			The size of the phantom (in units)
 * @param type			Type of the phantom
 * @param edgeWidth		How wide the edges should be, 0.0 indicates sharp edges, higher values causes the phantom to be more "derivable".
 */
Eigen::ArrayXXd UTILITIES_EXPORT phantom(Eigen::Vector2i size,
                                         PhantomType type = PhantomType::modifiedSheppLogan,
                                         double edgeWidth = 0.0);
}