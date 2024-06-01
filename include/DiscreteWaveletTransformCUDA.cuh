#ifndef DWT_CUH
#define DWT_CUH

#include "typedefs.hpp"
#include <array>


template <unsigned long matrix_size>
auto dwtCU(Typedefs::vec &signal, bool is_inverse, const std::array<double, matrix_size> &transform_matrix, const std::array<double, matrix_size> &inverse_matrix, const int user_levels) -> void;

#endif