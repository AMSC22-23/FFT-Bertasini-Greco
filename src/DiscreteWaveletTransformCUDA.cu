#include "DiscreteWaveletTransformCUDA.cuh"

//CUDA implementation
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void transformKernel(double* signal, const double* t_mat, const double* temp, int sub_step, int sub_size, const size_t matrix_size) {
    int j = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (j >= sub_size) return;
    int index_signal = j*sub_step;
    signal[index_signal] = 0;
    signal[index_signal+sub_step] = 0;
    for (unsigned long m=0; m < matrix_size; m+=1){
        signal[index_signal]              += temp[j+m]*t_mat[m];
        signal[index_signal+sub_step]     += temp[j+m]*t_mat[m+ matrix_size];
    }
}

auto cudabackend::dwtCU(Typedefs::vec &signal, const bool is_inverse, const std::span<const Typedefs::DType> &transform_matrix, const std::span<const Typedefs::DType> &inverse_matrix, const uint8_t user_levels) -> void
{
    auto& t_mat = is_inverse ? inverse_matrix : transform_matrix;
    const unsigned long matrix_size = t_mat.size() / 2;

    Typedefs::vec temp;
    int levels = user_levels == 0 ? log2(signal.size()) : user_levels;
    int start = is_inverse ? levels-1 : 0; 
    int end = is_inverse ? -1 : levels;
    int step = is_inverse ? -1 : 1;

    DType* d_signal;
    DType* d_t_mat;
    DType* d_temp;

    CUDA_CALL(cudaMalloc((void**)&d_signal, signal.size() * sizeof(DType)));
    CUDA_CALL(cudaMalloc((void**)&d_t_mat, t_mat.size() * sizeof(DType)));

    CUDA_CALL(cudaMemcpy(d_signal, signal.data(), signal.size() * sizeof(DType), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_t_mat, t_mat.data(), t_mat.size() * sizeof(DType), cudaMemcpyHostToDevice));

    for (int i = start; i != end; i += step) {
        temp.clear();
        int sub_step = pow(2, i);
        int sub_size = signal.size()/sub_step;

        for (int j = 0; j < sub_size; j++) temp.push_back(signal[j*sub_step]);
    
        if (!is_inverse) for (unsigned long j = 0; j < matrix_size-2; j++) temp.push_back(temp[j]);
        else             for (unsigned long j = 0; j < matrix_size-2; j++) temp.insert(temp.begin(), *(temp.end()-1-j));

        CUDA_CALL(cudaMalloc((void**)&d_temp, temp.size() * sizeof(DType)));
        CUDA_CALL(cudaMemcpy(d_temp, temp.data(), temp.size() * sizeof(DType), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (sub_size + threadsPerBlock - 1) / threadsPerBlock;
        transformKernel<<<blocksPerGrid, threadsPerBlock>>>(d_signal, d_t_mat, d_temp, sub_step, sub_size, matrix_size);
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaFree(d_temp));
        CUDA_CALL(cudaMemcpy(signal.data(), d_signal, signal.size() * sizeof(DType), cudaMemcpyDeviceToHost));
    }

    CUDA_CALL(cudaMemcpy(signal.data(), d_signal, signal.size() * sizeof(DType), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_signal));
    CUDA_CALL(cudaFree(d_t_mat));
}