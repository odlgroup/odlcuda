#include <algorithm>
#include <memory>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// thrust
#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/adjacent_difference.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

// ODL
#include <odlpp/cuda/DeviceVectorImpl.h>
#include <odlpp/cuda/TypeMacro.h>

// Utils
#include <odl_cpp_utils/cuda/thrustUtils.h>
#include <odl_cpp_utils/utils/cast.h>
#include <odl_cpp_utils/cuda/cutil_math.h>
#include <odl_cpp_utils/cuda/errcheck.h>

// Reductions
float sumImpl(const DeviceVector<float>& v) {
    return thrust::reduce(v.begin(), v.end());
}

__global__ void convKernel(const float* source, const float* kernel,
                           float* target, const int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= len) return;

    float value = 0.0f;

    for (int i = 0; i < len; i++) {
        value += source[i] *
                 kernel[(len + len / 2 + idx - i) % len]; // Positive modulo
    }

    target[idx] = value;
}

void convImpl(const DeviceVector<float>& source,
              const DeviceVector<float>& kernel, DeviceVector<float>& target) {
    size_t len = source.size();
    unsigned dimBlock = 256;
    unsigned dimGrid = narrow_cast<unsigned>(1 + (len / dimBlock));

    convKernel<<<dimGrid, dimBlock>>>(source.data(), kernel.data(), target.data(), narrow_cast<int>(len));
    CUDA_KERNEL_ERRCHECK;
}

__global__ void forwardDifferenceKernel(const int len, const float* source,
                                        float* target) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < len - 1;
         idx += blockDim.x * gridDim.x) {
        target[idx] = source[idx + 1] - source[idx];
    }
}
void forwardDifferenceImpl(const DeviceVector<float>& source,
                           DeviceVector<float>& target) {
    size_t len = source.size();
    unsigned dimBlock(256);
    unsigned dimGrid(
        std::min<unsigned>(128, narrow_cast<unsigned>(1 + (len / dimBlock))));

    forwardDifferenceKernel<<<dimBlock, dimGrid>>>(narrow_cast<int>(len), source.data(), target.data());
    CUDA_KERNEL_ERRCHECK;
}

__global__ void forwardDifferenceAdjointKernel(const int len,
                                               const float* source,
                                               float* target) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < len - 1;
         idx += blockDim.x * gridDim.x) {
        target[idx] = -source[idx] + source[idx - 1];
    }
}
void forwardDifferenceAdjointImpl(const DeviceVector<float>& source,
                                  DeviceVector<float>& target) {
    size_t len = source.size();
    unsigned dimBlock(256);
    unsigned dimGrid(
        std::min<unsigned>(128u, narrow_cast<unsigned>(1 + (len / dimBlock))));

    forwardDifferenceAdjointKernel<<<dimBlock, dimGrid>>>(narrow_cast<int>(len), source.data(), target.data());
    CUDA_KERNEL_ERRCHECK;
}

void maxVectorVectorImpl(const DeviceVector<float>& v1,
                         const DeviceVector<float>& v2,
                         DeviceVector<float>& target) {
    thrust::transform(v1.begin(), v1.end(), v2.begin(), target.begin(),
                      thrust::maximum<float>{});
}

void maxVectorScalarImpl(const DeviceVector<float>& source, float scalar,
                         DeviceVector<float>& target) {
    auto scalarIter = thrust::make_constant_iterator(scalar);
    thrust::transform(source.begin(), source.end(), scalarIter, target.begin(),
                      thrust::maximum<float>{});
}

struct DivideFunctor {
    __host__ __device__ float operator()(const float& dividend,
                                         const float& divisor) {
        return divisor != 0.0f ? dividend / divisor : 0.0f;
    }
};
void divideVectorVectorImpl(const DeviceVector<float>& dividend,
                            const DeviceVector<float>& divisor,
                            DeviceVector<float>& quotient) {
    thrust::transform(dividend.begin(), dividend.end(), divisor.begin(),
                      quotient.begin(), DivideFunctor{});
}

void addScalarImpl(const DeviceVector<float>& source, float scalar,
                   DeviceVector<float>& target) {
    auto scalarIter = thrust::make_constant_iterator(scalar);
    thrust::transform(source.begin(), source.end(), scalarIter, target.begin(),
                      thrust::plus<float>{});
}

__global__ void forwardDifference2DKernel(const int cols, const int rows,
                                          const float* data, float* dx,
                                          float* dy) {
    for (auto idy = blockIdx.y * blockDim.y + threadIdx.y + 1; idy < cols - 1;
         idy += blockDim.y * gridDim.y) {
        for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
             idx < rows - 1; idx += blockDim.x * gridDim.x) {
            const auto index = idx + rows * idy;

            dx[index] = data[index + 1] - data[index];
            dy[index] = data[index + rows] - data[index];
        }
    }
}

void forwardDifference2DImpl(const DeviceVector<float>& source,
                             DeviceVector<float>& dx, DeviceVector<float>& dy,
                             const int cols, const int rows) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(32, 32);

    forwardDifference2DKernel<<<dimGrid, dimBlock>>>(cols, rows, source.data(), dx.data(), dy.data());
    CUDA_KERNEL_ERRCHECK;
}

__global__ void forwardDifference2DAdjointKernel(const int cols, const int rows,
                                                 const float* dx,
                                                 const float* dy,
                                                 float* target) {
    for (auto idy = blockIdx.y * blockDim.y + threadIdx.y + 1; idy < cols - 1;
         idy += blockDim.y * gridDim.y) {
        for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
             idx < rows - 1; idx += blockDim.x * gridDim.x) {
            const auto index = idx + rows * idy;

            target[index] =
                -dx[index] + dx[index - 1] - dy[index] + dy[index - rows];
        }
    }
}

void forwardDifference2DAdjointImpl(const DeviceVector<float>& dx,
                                    const DeviceVector<float>& dy,
                                    DeviceVector<float>& target, const int cols,
                                    const int rows) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(32, 32);

    forwardDifference2DAdjointKernel<<<dimGrid, dimBlock>>>(cols, rows, dx.data(), dy.data(), target.data());
    CUDA_KERNEL_ERRCHECK;
}

#define PI 3.141592653589793f

__global__ void gaussianBlurXkernel(const float* source, float* target,
                                    const uint2 imageSize, float sigma_x,
                                    int kernel_width) {
    uint2 id{blockIdx.x * blockDim.x + threadIdx.x,
             blockIdx.y * blockDim.y + threadIdx.y};

    if (id.x >= imageSize.x || id.y >= imageSize.y) return;

    const float a = 1.0f / (sigma_x * sqrtf(2.0f * PI));
    const float b = 2.0f * sigma_x * sigma_x;

    float value = 0.0f;

    for (int i = id.x - kernel_width; i < id.x + kernel_width; i++) {
        int x = i;
        if (x < 0) x = -x;
        if (x >= imageSize.x) x = 2 * imageSize.x - x - 1;

        float dx = static_cast<float>(i - id.x);
        float fx = a * expf(-dx * dx / b);
        int point_id = x + imageSize.x * id.y;
        value += fx * source[point_id];
    }

    unsigned idx = id.x + imageSize.x * id.y;
    target[idx] = value;
}

__global__ void gaussianBlurYkernel(const float* source, float* target,
                                    const uint2 imageSize, float sigma_y,
                                    int kernel_height) {
    uint2 id{blockIdx.x * blockDim.x + threadIdx.x,
             blockIdx.y * blockDim.y + threadIdx.y};

    if (id.x >= imageSize.x || id.y >= imageSize.y) return;

    const float a = 1.0f / (sigma_y * sqrtf(2.0f * PI));
    const float b = 2.0f * sigma_y * sigma_y;

    float value = 0.0f;

    for (int i = id.y - kernel_height; i < id.y + kernel_height; i++) {
        int y = i;
        if (y < 0) y = -y;
        if (y >= imageSize.y) y = 2 * imageSize.y - y - 1;

        float dx = static_cast<float>(i - id.y);
        float fx = a * expf(-dx * dx / b);
        int point_id = id.x + imageSize.x * y;
        value += fx * source[point_id];
    }

    unsigned idx = id.x + imageSize.x * id.y;
    target[idx] = value;
}

void gaussianBlurImpl(const DeviceVector<float>& image,
                      DeviceVector<float>& temporary, DeviceVector<float>& out,
                      const int image_width, const int image_height,
                      const float sigma_x, const float sigma_y,
                      const int kernel_width, const int kernel_height) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(narrow_cast<unsigned>(1 + (image_width / dimBlock.x)),
                 narrow_cast<unsigned>(1 + (image_height / dimBlock.y)));

    gaussianBlurXkernel<<<dimGrid, dimBlock>>>(image.data(),
                                               temporary.data(),
                                               uint2{narrow_cast<unsigned>(image_width),
                                                     narrow_cast<unsigned>(image_height)},
                                               sigma_x, kernel_width);
    CUDA_KERNEL_ERRCHECK;
    gaussianBlurYkernel<<<dimGrid, dimBlock>>>(temporary.data(),
                                               out.data(),
                                               uint2{narrow_cast<unsigned>(image_width),
                                                     narrow_cast<unsigned>(image_height)},
                                               sigma_y,
                                               kernel_height);
    CUDA_KERNEL_ERRCHECK;
}
