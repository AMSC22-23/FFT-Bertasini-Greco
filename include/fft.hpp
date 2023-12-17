#ifndef FFT_HPP
#define FFT_HPP

#include <typedefs.hpp>

//@note Why not make a class for the fft? So that the different implementations
//      can be used interchangeably. You could have usied inheritance for instance·
namespace recursive { auto fft (vcpx&, const bool, const int) -> void; }

#endif