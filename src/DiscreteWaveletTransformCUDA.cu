#include "DiscreteWaveletTransformCUDA.cuh"

//CUDA implementation
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cuda_runtime.h>

namespace Typedefs {
    using vec = std::vector<double>;
}

#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

template <unsigned long matrix_size>
__global__ void transformKernel(double* signal, const double* t_mat, const double* temp, int sub_step, int sub_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= sub_size / 2) return;

    int index_signal = j * 2 * sub_step;
    signal[index_signal] = 0;
    signal[index_signal + sub_step] = 0;
    for (unsigned long m = 0; m < matrix_size*2; ++m) {
        signal[index_signal] += temp[j * 2 + m] * t_mat[m];
        signal[index_signal + sub_step] += temp[j * 2 + m] * t_mat[m + matrix_size*2];
    }
}

template <unsigned long matrix_size>
auto dwtCU(Typedefs::vec &signal, bool is_inverse, const std::array<double, matrix_size*2> &transform_matrix, const std::array<double, matrix_size*2> &inverse_matrix, const int user_levels) -> void {
    const auto& t_mat = is_inverse ? inverse_matrix : transform_matrix;
    int levels = user_levels == 0 ? log2(signal.size()) : user_levels;
    int start = is_inverse ? levels - 1 : 0;
    int end = is_inverse ? -1 : levels;
    int step = is_inverse ? -1 : 1;

    double* d_signal;
    double* d_t_mat;
    double* d_temp;
    size_t signal_size = signal.size() * sizeof(double);
    size_t t_mat_size = matrix_size * 2 * sizeof(double);  // Since t_mat contains two matrix_size parts

    CUDA_CALL(cudaMalloc((void**)&d_signal, signal_size));
    CUDA_CALL(cudaMalloc((void**)&d_t_mat, t_mat_size));

    CUDA_CALL(cudaMemcpy(d_signal, signal.data(), signal_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_t_mat, t_mat.data(), t_mat_size, cudaMemcpyHostToDevice));

    for (int i = start; i != end; i += step) {
        int sub_step = pow(2, i);
        int sub_size = signal.size() / sub_step;
        std::vector<double> temp;

        for (int j = 0; j < sub_size; j++) temp.push_back(signal[j * sub_step]);

        if (!is_inverse) {
            for (unsigned long j = 0; j < matrix_size*2 - 2; j++) temp.push_back(temp[j]);
        } else {
            for (unsigned long j = 0; j < matrix_size*2 - 2; j++) temp.insert(temp.begin(), *(temp.end() - 1 - j));
        }

        CUDA_CALL(cudaMalloc((void**)&d_temp, temp.size() * sizeof(double)));
        CUDA_CALL(cudaMemcpy(d_temp, temp.data(), temp.size() * sizeof(double), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (sub_size / 2 + threadsPerBlock - 1) / threadsPerBlock;
        transformKernel<matrix_size><<<blocksPerGrid, threadsPerBlock>>>(d_signal, d_t_mat, d_temp, sub_step, sub_size);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    CUDA_CALL(cudaMemcpy(signal.data(), d_signal, signal_size, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_signal));
    CUDA_CALL(cudaFree(d_t_mat));
    CUDA_CALL(cudaFree(d_temp));
}
    

template void dwtCU<2> (Typedefs::vec&, bool, const std::array<double, 4>&, const std::array<double, 4>&, const int);
template void dwtCU<4> (Typedefs::vec&, bool, const std::array<double, 8>&, const std::array<double, 8>&, const int);
template void dwtCU<6> (Typedefs::vec&, bool, const std::array<double, 12>&, const std::array<double, 12>&, const int);
template void dwtCU<8> (Typedefs::vec&, bool, const std::array<double, 16>&, const std::array<double, 16>&, const int);
template void dwtCU<10>(Typedefs::vec&, bool, const std::array<double, 20>&, const std::array<double, 20>&, const int);
template void dwtCU<16>(Typedefs::vec&, bool, const std::array<double, 32>&, const std::array<double, 32>&, const int);
template void dwtCU<20>(Typedefs::vec&, bool, const std::array<double, 40>&, const std::array<double, 40>&, const int);
template void dwtCU<30>(Typedefs::vec&, bool, const std::array<double, 60>&, const std::array<double, 60>&, const int);
template void dwtCU<40>(Typedefs::vec&, bool, const std::array<double, 80>&, const std::array<double, 80>&, const int);
