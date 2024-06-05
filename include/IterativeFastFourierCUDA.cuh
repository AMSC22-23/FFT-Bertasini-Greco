/**
* @file IterativeFastFourierCUDA.cuh
* @brief Declares a function for performing the Fast Fourier Transform on a complex vector using CUDA.
*/

#ifndef IFFT_CUH
#define IFFT_CUH

#include "typedefs.hpp"

/**
* @brief Perform the Fast Fourier Transform on a complex vector using CUDA.
* @param x The input complex vector.
* @param is_inverse A flag indicating whether to perform the inverse transform.
*/
auto fftCU(Typedefs::vcpx& x, const bool is_inverse) -> void;

#endif