/**
 * @file bitreverse.hpp
 * @brief Declares functions for bit-reversing vectors and 3D vectors (images).
 */

#ifndef BITREVERSE_HPP
#define BITREVERSE_HPP

#include <cstddef>
#include <cstdint>

#include "typedefs.hpp"

namespace tr::bitreverse
{

    /**
     * @brief Perform partial bit-reversal on a signal vector.
     * @param signal The signal vector to be bit-reversed.
     * @param n The size of the signal vector.
     * @param levels The number of levels for bit-reversal.
     */
    auto partial_bit_reverse(Typedefs::vec &signal, size_t n, uint8_t levels) -> void;

    /**
     * @brief Perform bit-reversal on a vector by creating a new bit-reversed copy.
     * @tparam T The data type of the vector elements.
     * @param v The input vector.
     */
    template <typename T>
    auto bit_reverse_copy(T &v) -> void;

    /**
     * @brief Find the next multiple of a given value that is greater than or equal to a given number.
     * @param n The number for which the next multiple needs to be found.
     * @param m The value with respect to which the multiple needs to be found.
     * @return The next multiple of m that is greater than or equal to n.
     */
    auto next_multiple_of_levels(size_t n, size_t m) -> size_t;

    /**
     * @brief Perform bit-reversal on a 3D vector (image).
     * @param image The input image as a 3D vector.
     * @param user_levels The number of levels for bit-reversal.
     */
    auto bit_reverse_image(Typedefs::vec3D &image, uint8_t user_levels) -> void;

    /**
     * @brief Reverse the bit-reversal operation on a 3D vector (image).
     * @param bit_reversed_image The bit-reversed image as a 3D vector.
     * @param levels The number of levels used for bit-reversal.
     */
    auto reverse_bit_reverse_image(Typedefs::vec3D &bit_reversed_image, uint8_t levels) -> void;
}
#endif