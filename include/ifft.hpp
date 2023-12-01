#ifndef INVERSE_TRANSFORM_HPP
#define INVERSE_TRANSFORM_HPP

#include <functional>
#include <typedefs.hpp>

auto ifft(vcpx X, const std::function<auto (vcpx&) -> void>& fft) -> vcpx;

#endif