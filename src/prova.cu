#include "prova.cuh"

__global__ void testKernel() {
  printf("Hello, CUDA!\n");
}

auto prova()  -> void {
  testKernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}