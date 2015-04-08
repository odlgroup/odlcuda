#pragma once
#include <algorithm>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// thrust
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/adjacent_difference.h>

// RL
#include <RLcpp/thrustUtils.h>

template <typename T>
struct uninitialized_allocator
    : thrust::device_malloc_allocator<T> {
    // note that construct is annotated as
    // a __host__ __device__ function
    __host__ __device__ void construct(T* p) {
        // no-op
    }
};

typedef thrust::device_vector<float> device_vector;
typedef std::shared_ptr<device_vector> device_vector_ptr;

device_vector_ptr makeThrustVector(size_t size) {
    return std::make_shared<device_vector>(size);
}

device_vector_ptr makeThrustVector(size_t size, float value) {
    return std::make_shared<device_vector>(size, value);
}

void linCombImpl(float a, const device_vector_ptr& x, float b, device_vector_ptr& y) {
    using namespace thrust::placeholders;

#if 1 //Efficient
    if (a == 0.0f) {
        if (b == 0.0f) { // y = 0
            thrust::fill(y->begin(), y->end(), 0.0f);
        } else if (b == 1.0f) {  // y = y, no-op
        } else if (b == -1.0f) { // y = -y
            thrust::transform(y->begin(), y->end(), y->begin(), -_1);
        } else { // y = b*y
            thrust::transform(y->begin(), y->end(), y->begin(), b * _1);
        }
    } else if (a == 1.0f) {
        if (b == 0.0f) { // y = x
            thrust::copy(x->begin(), x->end(), y->begin());
        } else if (b == 1.0f) { // y = x+y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), _1 + _2);
        } else if (b == -1.0f) { // y = x-y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), _1 - _2);
        } else { // y = x + b*y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), _1 + b * _2);
        }
    } else if (a == -1.0f) {
        if (b == 0.0f) { // y = -x
            thrust::transform(x->begin(), x->end(), y->begin(), -_1);
        } else if (b == 1.0f) { // y = -x+y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), -_1 + _2);
        } else if (b == -1.0f) { // y = -x-y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), -_1 - _2);
        } else { // y = -x + b*y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), -_1 + b * _2);
        }
    } else {
        if (b == 0.0f) { // y = a*x
            thrust::transform(x->begin(), x->end(), y->begin(), a * _1);
        } else if (b == 1.0f) { // y = a*x+y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), a * _1 + _2);
        } else if (b == -1.0f) { // y = a*x-y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), a * _1 - _2);
        } else { // y = a*x + b*y
            thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), a * _1 + b * _2);
        }
    }
#else //Basic
    thrust::transform(x->begin(), x->end(), y->begin(), y->begin(), a * _1 + b * _2);
#endif
}

void multiplyImpl(const device_vector_ptr& v1, device_vector_ptr& v2) {
    using namespace thrust::placeholders;
    thrust::transform(v1->begin(), v1->end(), v2->begin(), v2->begin(), _1 * _2);
}

float innerImpl(const device_vector_ptr& v1, const device_vector_ptr& v2) {
    return thrust::inner_product(v1->begin(), v1->end(), v2->begin(), 0.0f);
}

//Reductions
float sumImpl(const device_vector_ptr& v) {
    return thrust::reduce(v->begin(), v->end());
}

struct Square {
    __host__ __device__ float operator()(const float& x) const { return x * x; }
};
float normSqImpl(const device_vector_ptr& v1) {
    return thrust::transform_reduce(v1->begin(), v1->end(), Square{}, 0.0f, thrust::plus<float>{});
}

//Copies
void copyHostToDevice(double* source, device_vector_ptr& target) {
    thrust::copy_n(source, target->size(), target->begin());
}

void copyDeviceToHost(const device_vector_ptr& source, double* target) {
    thrust::copy(source->begin(), source->end(), target);
}

void printData(const device_vector_ptr& v1, std::ostream_iterator<float>& out) {
    thrust::copy(v1->begin(), v1->end(), out);
}

float getItemImpl(const device_vector_ptr& v1, int index) {
    return v1->operator[](index);
}

void setItemImpl(device_vector_ptr& v1, int index, float value) {
    v1->operator[](index) = value;
}

void getSliceImpl(const device_vector_ptr& v1, int start, int stop, int step, double* target) {
    if (step != 1) {
		auto iter = make_strided_range(v1->begin() + start, v1->begin() + stop, step);
        thrust::copy(iter.begin(), iter.end(), target);
    } else {
        thrust::copy(v1->begin() + start, v1->begin() + stop, target);
    }
}

void setSliceImpl(const device_vector_ptr& v1, int start, int stop, int step, double* source, int num) {
    if (step != 1) {
        auto iter = make_strided_range(v1->begin() + start, v1->begin() + stop, step);
        thrust::copy(source, source + num, iter.begin());
    } else {
        thrust::copy(source, source + num, v1->begin() + start);
    }
}

__global__ void convKernel(const float* source,
                           const float* kernel,
                           float* target,
                           int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= len)
        return;

    float value = 0.0f;

    for (int i = 0; i < len; i++) {
        value += source[i] * kernel[(len + len / 2 + idx - i) % len]; //Positive modulo
    }

    target[idx] = value;
}

void convImpl(const device_vector_ptr& source, const device_vector_ptr& kernel, device_vector_ptr& target) {
    int len = source->size();
    unsigned dimBlock(256);
    unsigned dimGrid(1 + (len / dimBlock));

    convKernel<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(source->data()),
                                      thrust::raw_pointer_cast(kernel->data()),
                                      thrust::raw_pointer_cast(target->data()),
                                      len);
}

// Functions
struct AbsoluteValue {
    __host__ __device__ float operator()(const float& f) { return fabs(f); }
};
void absImpl(const device_vector_ptr& source, device_vector_ptr& target) {
    thrust::transform(source->begin(), source->end(), target->begin(), AbsoluteValue{});
}

__global__ void forwardDifferenceKernel(const int len, const float* source, float* target) {
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < len - 1; idx += blockDim.x * gridDim.x) {
        target[idx] = source[idx + 1] - source[idx];
    }
}
void forwardDifferenceImpl(const device_vector_ptr& source, device_vector_ptr& target) {
    int len = source->size();
    unsigned dimBlock(256);
    unsigned dimGrid(std::min(128u, 1 + (len / dimBlock)));

    forwardDifferenceKernel<<<dimBlock, dimGrid>>>(len,
                                                   thrust::raw_pointer_cast(source->data()),
                                                   thrust::raw_pointer_cast(target->data()));
}

__global__ void forwardDifferenceAdjointKernel(const int len, const float* source, float* target) {
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < len - 1; idx += blockDim.x * gridDim.x) {
        target[idx] = -source[idx] + source[idx - 1];
    }
}
void forwardDifferenceAdjointImpl(const device_vector_ptr& source, device_vector_ptr& target) {
    int len = source->size();
    unsigned dimBlock(256);
    unsigned dimGrid(std::min(128u, 1 + (len / dimBlock)));

    forwardDifferenceAdjointKernel<<<dimBlock, dimGrid>>>(len,
                                                          thrust::raw_pointer_cast(source->data()),
                                                          thrust::raw_pointer_cast(target->data()));
}

void maxVectorVectorImpl(const device_vector_ptr& source, device_vector_ptr& target) {
    thrust::transform(source->begin(), source->end(), target->begin(), target->begin(), thrust::maximum<float>());
}

void maxVectorScalarImpl(const device_vector_ptr& source, float scalar, device_vector_ptr& target) {
    auto scalarIter = thrust::make_constant_iterator(scalar);
    thrust::transform(source->begin(), source->end(), scalarIter, target->begin(), thrust::maximum<float>());
}

void addScalarImpl(const device_vector_ptr& source, float scalar, device_vector_ptr& target) {
    using namespace thrust::placeholders;
    thrust::transform(source->begin(), source->end(), target->begin(), _1 + scalar);
}

struct SignFunctor {
    __host__ __device__ float operator()(const float& f) { return (0.0f < f) - (f < 0.0f); }
};
void signImpl(const device_vector_ptr& source, device_vector_ptr& target) {
    thrust::transform(source->begin(), source->end(), target->begin(), SignFunctor());
}

__global__ void forwardDifference2DKernel(const int cols, const int rows, const float* data, float* dx, float* dy) {
    for (auto idy = blockIdx.y * blockDim.y + threadIdx.y + 1; idy < rows - 1; idy += blockDim.y * gridDim.y) {
        for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < cols - 1; idx += blockDim.x * gridDim.x) {
            const auto index = idx + cols * idy;

            dx[index] = data[index + 1] - data[index];
            dy[index] = data[index + cols] - data[index];
        }
    }
}
void forwardDifference2DImpl(const device_vector_ptr& source, device_vector_ptr& dx, device_vector_ptr& dy, const int cols, const int rows) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(32, 32);

    forwardDifference2DKernel<<<dimGrid, dimBlock>>>(cols, rows,
                                                     thrust::raw_pointer_cast(source->data()),
                                                     thrust::raw_pointer_cast(dx->data()),
                                                     thrust::raw_pointer_cast(dy->data()));
}

__global__ void forwardDifference2DAdjointKernel(const int cols, const int rows, const float* dx, const float* dy, float* target) {
    for (auto idy = blockIdx.y * blockDim.y + threadIdx.y + 1; idy < rows - 1; idy += blockDim.y * gridDim.y) {
        for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < cols - 1; idx += blockDim.x * gridDim.x) {
            const auto index = idx + cols * idy;

            target[index] = -dx[index] + dx[index - 1] - dy[index] + dy[index - cols];
        }
    }
}

void forwardDifference2DAdjointImpl(const device_vector_ptr& dx, const device_vector_ptr& dy, device_vector_ptr& target, const int cols, const int rows) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(32, 32);

    forwardDifference2DAdjointKernel<<<dimGrid, dimBlock>>>(cols, rows,
                                                            thrust::raw_pointer_cast(dx->data()),
                                                            thrust::raw_pointer_cast(dy->data()),
                                                            thrust::raw_pointer_cast(target->data()));
}