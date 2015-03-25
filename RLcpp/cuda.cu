#pragma once
// includes CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// thrust
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

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

typedef thrust::device_vector<float, uninitialized_allocator<float>> device_vector;
typedef std::shared_ptr<device_vector> device_vector_ptr;

device_vector_ptr makeThrustVector(size_t size) {
    return std::make_shared<device_vector>(size);
}

device_vector_ptr makeThrustVector(size_t size, float value) {
    return std::make_shared<device_vector>(size, value);
}

struct lincombFunctor {
  public:
    lincombFunctor(float a, float b) : _a(a),
                                       _b(b) {
    }
    __host__ __device__ float operator()(const float f1, const float f2) const {
        return _a * f1 + _b * f2;
    }

  private:
    float _a;
    float _b;
};

void linCombImpl(float a, const device_vector_ptr& v1, float b, device_vector_ptr& v2) {
    thrust::transform(v1->begin(), v1->end(), v2->begin(), v2->begin(), lincombFunctor(a, b));
}

void copyHostToDevice(double* source, device_vector_ptr& target) {
    thrust::copy_n(source, target->size(), target->begin());
}

void copyDeviceToHost(const device_vector_ptr& source, double* target) {
    thrust::copy(source->begin(), source->end(), target);
}

void copyDeviceToDevice(const device_vector_ptr& source, device_vector_ptr& target) {
    thrust::copy(source->begin(), source->end(), target->begin());
}

float innerProduct(const device_vector_ptr& v1, const device_vector_ptr& v2) {
    return thrust::inner_product(v1->begin(), v1->end(), v2->begin(), 0.0f);
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
    strided_range<device_vector::iterator> iter(v1->begin() + start, v1->begin() + stop, step);
    thrust::copy(iter.begin(), iter.end(), target);
}

void setSliceImpl(const device_vector_ptr& v1, int start, int stop, int step, double* source, int num) {
    strided_range<device_vector::iterator> iter(v1->begin() + start, v1->begin() + stop, step);
    thrust::copy(source, source + num, iter.begin());
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
        value += source[i] * kernel[(len + len/2 + idx - i) % len]; //Positive modulo
    }

    target[idx] = value;
}

void convImpl(const device_vector_ptr& source, const device_vector_ptr& kernel, device_vector_ptr& target) {
    int len = source->size();
    dim3 dimBlock(256, 1);
    dim3 dimGrid(static_cast<unsigned int>(1 + (len / dimBlock.x)), 1);

    convKernel << <dimGrid, dimBlock>>> (thrust::raw_pointer_cast(source->data()),
                                         thrust::raw_pointer_cast(kernel->data()),
                                         thrust::raw_pointer_cast(target->data()),
										 len);
}