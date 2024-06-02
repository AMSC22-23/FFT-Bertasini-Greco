#ifndef TIME_EV_HPP
#define TIME_EV_HPP

#include <chrono>

#include "FourierTransform.hpp"
#include "DiscreteWaveletTransform.hpp"

auto time_ev ( const Typedefs::vcpx& x, const std::unique_ptr<FourierTransform>& fft) -> long unsigned int;

auto time_ev_dwt (const Typedefs::vec& x, const DiscreteWaveletTransform& dwt) -> long unsigned int;

#endif