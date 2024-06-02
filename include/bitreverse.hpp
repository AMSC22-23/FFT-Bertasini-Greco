#ifndef BITREVERSE_HPP
#define BITREVERSE_HPP

#include <cstddef>
#include <cstdint>

#include "typedefs.hpp"

auto partial_bit_reverse(Typedefs::vec& signal, size_t n, uint8_t levels) -> void;
template <typename T> auto bit_reverse_copy(T& v) -> void;
auto next_multiple_of_levels(size_t n, size_t m) -> size_t;
auto bit_reverse_image(Typedefs::vec3D& image, uint8_t user_levels) -> void;
auto reverse_bit_reverse_image(Typedefs::vec3D &bit_reversed_image, uint8_t levels) -> void;

#endif