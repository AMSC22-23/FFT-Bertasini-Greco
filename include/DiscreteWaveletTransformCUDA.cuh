#ifndef DWT_CUH
#define DWT_CUH

#include <cstdint>
#include <span>

#include "typedefs.hpp"

auto dwtCU(Typedefs::vec &signal, const bool is_inverse, const std::span<const Typedefs::DType> &transform_matrix, const std::span<const Typedefs::DType> &inverse_matrix, const uint8_t user_levels) -> void;

#endif