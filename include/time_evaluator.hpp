/**
 * @file time_evaluator.hpp
 * @brief Declares functions for timing the execution of Fourier and wavelet transforms.
 */

#ifndef TIME_EV_HPP
#define TIME_EV_HPP

#include <chrono>

#include "DiscreteWaveletTransform.hpp"
#include "FourierTransform.hpp"

namespace tr::test_suite
{

    /**
     * @brief Time the execution of a Fourier transform on a complex signal.
     * @param x The input complex signal.
     * @param fft The Fourier transform object.
     * @return The execution time in microseconds.
     */
    auto time_ev(const Typedefs::vcpx &x, const std::unique_ptr<tr::FourierTransform> &fft) -> long unsigned int;

    /**
     * @brief Time the execution of a discrete wavelet transform on a signal vector.
     * @param x The input signal vector.
     * @param dwt The discrete wavelet transform object.
     * @return The execution time in microseconds.
     */
    auto time_ev_dwt(const Typedefs::vec &x, const tr::DiscreteWaveletTransform &dwt) -> long unsigned int;
}

#endif