/**
* @file utils.hpp
* @brief Declares utility functions for various operations.
*/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cstdint>

#include "typedefs.hpp"

/**
* @brief Find the next power of 2 that is greater than or equal to the given number.
* @param n The input number.
* @return The next power of 2 that is greater than or equal to n.
*/
auto next_power_of_2(const std::size_t n) -> std::size_t;

/**
* @brief Count the number of subdivisions for a given row and column index.
* @param i The row index.
* @param j The column index.
* @param rows The total number of rows.
* @param cols The total number of columns.
* @param subdivisions The number of subdivisions.
* @return The count of subdivisions for the given row and column index.
*/
auto countSubdivisions(unsigned int i, unsigned int j, const unsigned int rows, const unsigned int cols, const unsigned int subdivisions) -> int;

/**
* @brief Read a signal from a file and store it in a vector.
* @param signal_file The path to the signal file.
* @param real_signal The vector to store the read signal.
*/
auto read_signal(const std::string& signal_file, Typedefs::vec& real_signal) -> void;

#endif