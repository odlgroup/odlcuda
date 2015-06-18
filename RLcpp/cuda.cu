#include <algorithm>
#include <memory>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// thrust
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/adjacent_difference.h>

// RL
#include <RLcpp/thrustUtils.h>
#include <RLcpp/DeviceVector.h>

template <typename T>
class DeviceVector {
  public:
    virtual ~DeviceVector() {}
    virtual T* data() = 0;
    virtual T const* data() const = 0;
    virtual size_t size() const = 0;

    thrust::device_ptr<T> begin() {
        return thrust::device_pointer_cast<T>(data());
    }
    thrust::device_ptr<const T> begin() const {
        return thrust::device_pointer_cast<const T>(data());
    }

    thrust::device_ptr<T> end() {
        return begin() + size();
    }
    thrust::device_ptr<const T> end() const {
        return begin() + size();
    }

    thrust::device_reference<T> operator[](size_t index) {
        return thrust::device_reference<T>{begin() + index};
    }
    thrust::device_reference<const T> operator[](size_t index) const {
        return thrust::device_reference<const T>{begin() + index};
    }
};

template <typename T>
class ThrustDeviceVector : public DeviceVector<T> {
  private:
    thrust::device_vector<T> _data;

  public:
    ThrustDeviceVector(size_t size)
        : _data(size) {}

    ThrustDeviceVector(size_t size, T value)
        : _data(size, value) {}

    T* data() override {
        return thrust::raw_pointer_cast(_data.data());
    }
    T const* data() const override {
        return thrust::raw_pointer_cast(_data.data());
    }

    size_t size() const override {
        return _data.size();
    }
};

template <typename T>
class WrapperDeviceVector : public DeviceVector<T> {
  private:
    T * const _data;
    const size_t _size;

  public:
    WrapperDeviceVector(T * data, size_t size)
        : _data(data),
          _size(size) {}

    T* data() override {
        return _data;
    }

    T const* data() const override {
        return _data;
    }

    size_t size() const override {
        return _size;
    }
};

template <typename T>
DeviceVectorPtr<T> makeThrustVector(size_t size) {
    return std::make_shared<ThrustDeviceVector<T>>(size);
}
template DeviceVectorPtr<float> makeThrustVector(size_t size);
template DeviceVectorPtr<unsigned char> makeThrustVector(size_t size);

template <typename T>
DeviceVectorPtr<T> makeThrustVector(size_t size, T value) {
    return std::make_shared<ThrustDeviceVector<T>>(size, value);
}
template DeviceVectorPtr<float> makeThrustVector(size_t size, float value);
template DeviceVectorPtr<unsigned char> makeThrustVector(size_t size, unsigned char value);

template <typename T>
DeviceVectorPtr<T> makeWrapperVector(T * data, size_t size) {
    return std::make_shared<WrapperDeviceVector<T>>(data, size);
}
template DeviceVectorPtr<float> makeWrapperVector(float * data, size_t size);
template DeviceVectorPtr<unsigned char> makeWrapperVector(unsigned char * data, size_t size);

template <typename T>
T* getRawPtr(DeviceVector<T>& vec) {
	return vec.data();
}
template float* getRawPtr(DeviceVector<float>& vec);
template unsigned char* getRawPtr(DeviceVector<unsigned char>& vec);

template <typename T>
T const* getRawPtr(const DeviceVector<T>& vec) {
    return vec.data();
}
template const float * getRawPtr(const DeviceVector<float>& vec);
template const unsigned char * getRawPtr(const DeviceVector<unsigned char>& vec);


void printData(const DeviceVector<float>& v1, std::ostream_iterator<float>& out, int numel) {
	thrust::copy(v1.begin(), v1.begin() + numel, out);
}

template <typename T>
T getItemImpl(const DeviceVector<T>& v1, int index) {
	return v1[index];
}
template float getItemImpl(const DeviceVector<float>& v1, int index);
template unsigned char getItemImpl(const DeviceVector<unsigned char>& v1, int index);

template <typename T>
void setItemImpl(DeviceVector<T>& v1, int index, T value) {
	v1[index] = value;
}
template void setItemImpl(DeviceVector<float>& v1, int index, float value);
template void setItemImpl(DeviceVector<unsigned char>& v1, int index, unsigned char value);

template <typename I1, typename I2>
void stridedGetImpl(I1 fromBegin, I1 fromEnd, I2 toBegin, int step) {
	if (step == 1) {
		thrust::copy(fromBegin, fromEnd, toBegin);
	}
	else {
		auto iter = make_strided_range(fromBegin, fromEnd, step);
		thrust::copy(iter.begin(), iter.end(), toBegin);
	}
}

template <typename T, typename S>
void getSliceImpl(const DeviceVector<T>& v1, int start, int stop, int step, S* target) {
	if (step > 0) {
		stridedGetImpl(v1.begin() + start, v1.begin() + stop, target, step);
	}
	else {
		auto reversedBegin = thrust::make_reverse_iterator(v1.begin() + start);
		auto reversedEnd = thrust::make_reverse_iterator(v1.begin() + stop);

		stridedGetImpl(reversedBegin, reversedEnd, target, -step);
	}
}
template void getSliceImpl(const DeviceVector<float>& v1, int start, int stop, int step, double* target);
template void getSliceImpl(const DeviceVector<unsigned char>& v1, int start, int stop, int step, double* target);
template void getSliceImpl(const DeviceVector<unsigned char>& v1, int start, int stop, int step, unsigned char* target);

template <typename I1, typename I2>
void stridedSetImpl(I1 fromBegin, I1 fromEnd, I2 toBegin, I2 toEnd, int step) {
	if (step == 1) {
		thrust::copy(fromBegin, fromEnd, toBegin);
	}
	else {
		auto iter = make_strided_range(toBegin, toEnd, step);
		thrust::copy(fromBegin, fromEnd, iter.begin());
	}
}

template <typename T, typename S>
void setSliceImpl(DeviceVector<T>& v1, int start, int stop, int step, const S * source, int num) {
	if (step > 0) {
		stridedSetImpl(source, source + num, v1.begin() + start, v1.begin() + stop, step);
	}
	else {
		auto reversedBegin = thrust::make_reverse_iterator(v1.begin() + start);
		auto reversedEnd = thrust::make_reverse_iterator(v1.begin() + stop);

		stridedSetImpl(source, source + num, reversedBegin, reversedEnd, -step);
	}
}
template void setSliceImpl(DeviceVector<float>& v1, int start, int stop, int step, const double* source, int num);
template void setSliceImpl(DeviceVector<unsigned char>& v1, int start, int stop, int step, const double* source, int num);
template void setSliceImpl(DeviceVector<unsigned char>& v1, int start, int stop, int step, const unsigned char* source, int num);

void linCombImpl(DeviceVector<float>& z, float a, const DeviceVector<float>& x, float b, const DeviceVector<float>& y) {
    using namespace thrust::placeholders;

#if 1 //Efficient
    if (a == 0.0f) {
        if (b == 0.0f) { // z = 0
            thrust::fill(z.begin(), z.end(), 0.0f);
        } else if (b == 1.0f) { // z = y
            thrust::copy(y.begin(), y.end(), z.begin());
        } else if (b == -1.0f) { // y = -y
            thrust::transform(y.begin(), y.end(), z.begin(), -_1);
        } else { // y = b*y
            thrust::transform(y.begin(), y.end(), z.begin(), b * _1);
        }
    } else if (a == 1.0f) {
        if (b == 0.0f) { // z = x
            thrust::copy(x.begin(), x.end(), z.begin());
        } else if (b == 1.0f) { // z = x+y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), _1 + _2);
        } else if (b == -1.0f) { // z = x-y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), _1 - _2);
        } else { // z = x + b*y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), _1 + b * _2);
        }
    } else if (a == -1.0f) {
        if (b == 0.0f) { // z = -x
            thrust::transform(x.begin(), x.end(), z.begin(), -_1);
        } else if (b == 1.0f) { // z = -x+y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), -_1 + _2);
        } else if (b == -1.0f) { // z = -x-y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), -_1 - _2);
        } else { // z = -x + b*y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), -_1 + b * _2);
        }
    } else {
        if (b == 0.0f) { // z = a*x
            thrust::transform(x.begin(), x.end(), z.begin(), a * _1);
        } else if (b == 1.0f) { // z = a*x+y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), a * _1 + _2);
        } else if (b == -1.0f) { // z = a*x-y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), a * _1 - _2);
        } else { // z = a*x + b*y
            thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), a * _1 + b * _2);
        }
    }
#else //Basic
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), a * _1 + b * _2);
#endif
}

void multiplyImpl(const DeviceVector<float>& v1, DeviceVector<float>& v2) {
    using namespace thrust::placeholders;
    thrust::transform(v1.begin(), v1.end(), v2.begin(), v2.begin(), _1 * _2);
}

float innerImpl(const DeviceVector<float>& v1, const DeviceVector<float>& v2) {
    return thrust::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0f);
}

//Reductions
float sumImpl(const DeviceVector<float>& v) {
    return thrust::reduce(v.begin(), v.end());
}

struct Square {
    __host__ __device__ float operator()(const float& x) const { return x * x; }
};
float normImpl(const DeviceVector<float>& v1) {
    return sqrtf(thrust::transform_reduce(v1.begin(), v1.end(), Square{}, 0.0f, thrust::plus<float>{}));
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

void convImpl(const DeviceVector<float>& source, const DeviceVector<float>& kernel, DeviceVector<float>& target) {
    int len = source.size();
    unsigned dimBlock(256);
    unsigned dimGrid(1 + (len / dimBlock));

    convKernel<<<dimGrid, dimBlock>>>(source.data(),
                                      kernel.data(),
                                      target.data(),
                                      len);
}

// Functions
struct AbsoluteValueFunctor {
    __host__ __device__ float operator()(const float& f) { return fabs(f); }
};
void absImpl(const DeviceVector<float>& source, DeviceVector<float>& target) {
    thrust::transform(source.begin(), source.end(), target.begin(), AbsoluteValueFunctor{});
}

__global__ void forwardDifferenceKernel(const int len, const float* source, float* target) {
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < len - 1; idx += blockDim.x * gridDim.x) {
        target[idx] = source[idx + 1] - source[idx];
    }
}
void forwardDifferenceImpl(const DeviceVector<float>& source, DeviceVector<float>& target) {
    int len = source.size();
    unsigned dimBlock(256);
    unsigned dimGrid(std::min(128u, 1 + (len / dimBlock)));

    forwardDifferenceKernel<<<dimBlock, dimGrid>>>(len,
                                                   source.data(),
                                                   target.data());
}

__global__ void forwardDifferenceAdjointKernel(const int len, const float* source, float* target) {
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < len - 1; idx += blockDim.x * gridDim.x) {
        target[idx] = -source[idx] + source[idx - 1];
    }
}
void forwardDifferenceAdjointImpl(const DeviceVector<float>& source, DeviceVector<float>& target) {
    int len = source.size();
    unsigned dimBlock(256);
    unsigned dimGrid(std::min(128u, 1 + (len / dimBlock)));

    forwardDifferenceAdjointKernel<<<dimBlock, dimGrid>>>(len,
                                                          source.data(),
                                                          target.data());
}

void maxVectorVectorImpl(const DeviceVector<float>& v1, const DeviceVector<float>& v2, DeviceVector<float>& target) {
    thrust::transform(v1.begin(), v1.end(), v2.begin(), target.begin(), thrust::maximum<float>{});
}

void maxVectorScalarImpl(const DeviceVector<float>& source, float scalar, DeviceVector<float>& target) {
    auto scalarIter = thrust::make_constant_iterator(scalar);
    thrust::transform(source.begin(), source.end(), scalarIter, target.begin(), thrust::maximum<float>{});
}

struct DivideFunctor {
    __host__ __device__ float operator()(const float& dividend, const float& divisor) { return divisor != 0.0f ? dividend / divisor : 0.0f; }
};
void divideVectorVectorImpl(const DeviceVector<float>& dividend, const DeviceVector<float>& divisor, DeviceVector<float>& quotient) {
    thrust::transform(dividend.begin(), dividend.end(), divisor.begin(), quotient.begin(), DivideFunctor{});
}

void addScalarImpl(const DeviceVector<float>& source, float scalar, DeviceVector<float>& target) {
    auto scalarIter = thrust::make_constant_iterator(scalar);
    thrust::transform(source.begin(), source.end(), scalarIter, target.begin(), thrust::plus<float>{});
}

struct SignFunctor {
    __host__ __device__ float operator()(const float& f) { return (0.0f < f) - (f < 0.0f); }
};
void signImpl(const DeviceVector<float>& source, DeviceVector<float>& target) {
    thrust::transform(source.begin(), source.end(), target.begin(), SignFunctor{});
}
struct SqrtFunctor {
    __host__ __device__ float operator()(const float& f) { return f > 0.0f ? sqrtf(f) : 0.0f; }
};
void sqrtImpl(const DeviceVector<float>& source, DeviceVector<float>& target) {
    thrust::transform(source.begin(), source.end(), target.begin(), SqrtFunctor{});
}

__global__ void forwardDifference2DKernel(const int cols, const int rows, const float* data, float* dx, float* dy) {
    for (auto idy = blockIdx.y * blockDim.y + threadIdx.y + 1; idy < cols - 1; idy += blockDim.y * gridDim.y) {
        for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < rows - 1; idx += blockDim.x * gridDim.x) {
            const auto index = idx + rows * idy;

            dx[index] = data[index + 1] - data[index];
            dy[index] = data[index + rows] - data[index];
        }
    }
}

void forwardDifference2DImpl(const DeviceVector<float>& source, DeviceVector<float>& dx, DeviceVector<float>& dy, const int cols, const int rows) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(32, 32);

    forwardDifference2DKernel<<<dimGrid, dimBlock>>>(cols, rows,
                                                     source.data(),
                                                     dx.data(),
                                                     dy.data());
}

__global__ void forwardDifference2DAdjointKernel(const int cols, const int rows, const float* dx, const float* dy, float* target) {
    for (auto idy = blockIdx.y * blockDim.y + threadIdx.y + 1; idy < cols - 1; idy += blockDim.y * gridDim.y) {
        for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < rows - 1; idx += blockDim.x * gridDim.x) {
            const auto index = idx + rows * idy;

            target[index] = -dx[index] + dx[index - 1] - dy[index] + dy[index - rows];
        }
    }
}

void forwardDifference2DAdjointImpl(const DeviceVector<float>& dx, const DeviceVector<float>& dy, DeviceVector<float>& target, const int cols, const int rows) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(32, 32);

    forwardDifference2DAdjointKernel<<<dimGrid, dimBlock>>>(cols, rows,
                                                            dx.data(),
                                                            dy.data(),
                                                            target.data());
}
