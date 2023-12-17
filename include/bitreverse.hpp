#ifndef BITREVERSE_HPP
#define BITREVERSE_HPP

#include <typedefs.hpp>

//@note I see you like the -> version of function declarations, I like it too.
auto bit_reverse_copy(vcpx& v) -> void;
auto next_power_of_2(long unsigned int n) -> long unsigned int;

#endif