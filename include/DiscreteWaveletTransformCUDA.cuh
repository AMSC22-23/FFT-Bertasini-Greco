#ifndef DWT_CUH
#define DWT_CUH

#include "typedefs.hpp"
#include <span>

auto dwtCU(Typedefs::vec &signal, bool is_inverse, const std::span<const double> &transform_matrix, const std::span<const double> &inverse_matrix, const int user_levels) -> void;

#endif