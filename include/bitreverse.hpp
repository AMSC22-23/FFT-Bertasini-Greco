#ifndef BITREVERSE_HPP
#define BITREVERSE_HPP

#include <typedefs.hpp>

auto partial_bit_reverse(Typedefs::vec& signal, size_t n, uint8_t levels) -> void;
template <typename T> auto bit_reverse_copy(T& v) -> void;
auto next_power_of_2(size_t n) -> size_t;

#endif