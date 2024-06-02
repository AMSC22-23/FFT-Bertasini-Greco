#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cstdint>

#include "typedefs.hpp"

auto next_power_of_2(const std::size_t n) -> std::size_t;
auto countSubdivisions(unsigned int i, unsigned int j, const unsigned int size, const unsigned int subdivisions) -> int;
auto read_signal(const std::string& signal_file, Typedefs::vec& real_signal) -> void;

#endif