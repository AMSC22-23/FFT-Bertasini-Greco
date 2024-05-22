#ifndef BITREVERSE_HPP
#define BITREVERSE_HPP

#include <typedefs.hpp>

template <typename T> auto bit_reverse_copy(T& v) -> void;
auto next_power_of_2(size_t n) -> size_t;

#endif