#ifndef TIME_EV_HPP
#define TIME_EV_HPP

#include <typedefs.hpp>
#include <FourierTransform.hpp>
#include <chrono>

auto time_ev ( const vcpx&, const std::unique_ptr<FourierTransform>&) -> long unsigned int;

#endif