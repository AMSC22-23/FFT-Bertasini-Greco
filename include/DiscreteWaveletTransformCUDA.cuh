/**
* @file DiscreteWaveletTransformCUDA.cuh
* @brief Declares a function for performing the discrete wavelet transform on a signal vector using CUDA.
*/

#ifndef DWT_CUH
#define DWT_CUH

#include <cstdint>
#include <span>

#include "typedefs.hpp"

/**
* @brief Perform the discrete wavelet transform on a signal vector using CUDA.
* @param signal The input signal vector.
* @param is_inverse A flag indicating whether to perform the inverse transform.
* @param transform_matrix The transform matrix for the forward transform.
* @param inverse_matrix The transform matrix for the inverse transform.
* @param user_levels The number of levels to be used in the transform.
*/
auto dwtCU(Typedefs::vec& signal, const bool is_inverse, const std::span<const Typedefs::DType>& transform_matrix, const std::span<const Typedefs::DType>& inverse_matrix, const uint8_t user_levels) -> void;

#endif