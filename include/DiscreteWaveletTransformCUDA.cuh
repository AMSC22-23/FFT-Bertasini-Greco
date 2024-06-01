#ifndef DWT_CUH
#define DWT_CUH

#include "typedefs.hpp"
#include <array>


template <unsigned long matrix_size>
auto dwtCU(Typedefs::vec &signal, bool is_inverse, const std::array<double, matrix_size*2> &transform_matrix, const std::array<double, matrix_size*2> &inverse_matrix, const int user_levels) -> void;

#endif