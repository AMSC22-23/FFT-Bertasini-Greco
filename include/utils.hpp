#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cstdint>

#include "typedefs.hpp"

auto next_power_of_2(std::size_t n) -> std::size_t;
auto countSubdivisions(int i, int j, int size, int subdivisions) -> int;
auto read_signal(const std::string& signal_file, Typedefs::vec& real_signal) -> void;

#endif