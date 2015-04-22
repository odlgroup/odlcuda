#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace RLCpp {

/**
 * Calculates ray box intersection of box and a ray passing through x0 and x1.
 *
 * Behaviour when the ray only touches the edge is left undefined for efficiency.
 *
 * http://www.gamedev.net/topic/553635-ray-tracing---ray-aabb-intersection/
 */
template <typename T, int N>
bool intersects(const Eigen::AlignedBox<T, N>& box,
                const Eigen::Matrix<T, N, 1>& x0,
                const Eigen::Matrix<T, N, 1>& x1,
                T& t0,
                T& t1) {
    auto direction = (x1 - x0).normalized();

    t0 = T(0);
    t1 = std::numeric_limits<T>::max();

    for (int i = 0; i < N; ++i) {
        T min_t = (box.min()[i] - x0[i]) / direction[i];
        T max_t = (box.max()[i] - x0[i]) / direction[i];

        if (min_t > max_t) {
            t0 = std::max(max_t, t0);
            t1 = std::min(min_t, t1);
        } else {
            t0 = std::max(min_t, t0);
            t1 = std::min(max_t, t1);
        }

        if (t0 > t1)
            return false;
    }

    return true;
}

/**
 * Same as intersects(box,x0,x1,IGNORE,IGNORE)
 */
template <typename T, int N>
bool intersects(const Eigen::AlignedBox<T, N>& box,
                const Eigen::Matrix<T, N, 1>& x0,
                const Eigen::Matrix<T, N, 1>& x1) {
    T a, b; //Throwaway parameters
    return intersects(box, x0, x1, a, b);
}

/**
 * Equivalent of meshgrid in Matlab
 */
inline void meshgrid(const Eigen::VectorXd& x, const Eigen::VectorXd& y, Eigen::ArrayXXd& xMat, Eigen::ArrayXXd& yMat) {
    xMat = x.replicate(1, y.size()).array();
    yMat = y.transpose().replicate(x.size(), 1).array();
}

inline Eigen::Map<const Eigen::ArrayXXd> oneDtotwoD(const Eigen::VectorXd& x, const Eigen::Vector2i& size) {
	assert(size[0] * size[1] == x.size()); //Data sizes fit
    return Eigen::Map<const Eigen::ArrayXXd>(x.data(), size[0], size[1]);
}

inline Eigen::Map<const Eigen::VectorXd> twoDtooneD(const Eigen::ArrayXXd& data) {
    return Eigen::Map<const Eigen::VectorXd>(data.data(), data.size());
}
}
