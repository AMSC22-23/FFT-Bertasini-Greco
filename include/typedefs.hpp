#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <complex>
#include <vector>
#include <functional>

using cpx = std::complex<double>;
using vcpx = std::vector<cpx>;
using ft = std::function<auto (vcpx&, const bool, const int) -> void>;

#endif