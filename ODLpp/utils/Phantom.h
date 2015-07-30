#pragma once

#include <ODLpp/utils/utilities_export.h>
#include <ODLpp/utils/StandardPhantoms.h>

#include <Eigen/Dense>

namespace ODLpp {
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