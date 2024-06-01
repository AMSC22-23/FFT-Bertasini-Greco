#include "IterativeFastFourierCUDA.cuh"
#include "bitreverse.hpp"

#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <cuda/std/cmath>
#include <iostream>
#include <cmath>


// typedef cuda::std::complex<double> cpx;
using cpx = cuda::std::complex<double>;

__global__ void fft_kernel(cpx *x, int N, int m, int is_inverse) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N / m) {
        int k_m = k * m;
        cpx Wm = cuda::std::polar(1.0, (1-2*is_inverse)*-2*M_PI/m);
        cpx W = 1;
        for (int j = 0; j < m/2; j++) {
            cpx t = W * x[k_m + j + m/2];
            cpx u = x[k_m + j];
            x[k_m + j] = u + t;
            x[k_m + j + m/2] = u - t;
            W *= Wm;
        }
    }
}

auto fftCU (Typedefs::vcpx& x, const bool is_inverse) -> void{
    size_t N = x.size();
    if (N == 1) return;
    
    // Bit reverse copy
    bit_reverse_copy(x);

    cpx *d_x;
    cudaMalloc((void**)&d_x, N * sizeof(cpx));
    cudaMemcpy(d_x, x.data(), N * sizeof(cpx), cudaMemcpyHostToDevice);

    int blockSize = 256;
    for (size_t s = 1; s <= log2(N); s++) {
        size_t m = 1 << s;
        int gridSize = (N / m + blockSize - 1) / blockSize;
        fft_kernel<<<gridSize, blockSize>>>(d_x, N, m, is_inverse);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(x.data(), d_x, N * sizeof(cpx), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
}