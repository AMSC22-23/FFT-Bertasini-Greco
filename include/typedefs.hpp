#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <complex>
#include <vector>
#include <functional>


//@note it is better to use traits to collect type alias or at least
//      use a namespace to avoid polluting the global namespace.
using cpx = std::complex<double>;
using vcpx = std::vector<cpx>;
using ft = std::function<auto (vcpx&, const bool, const int) -> void>;

#endif