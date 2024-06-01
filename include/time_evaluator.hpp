#ifndef TIME_EV_HPP
#define TIME_EV_HPP

#include <typedefs.hpp>
#include <FourierTransform.hpp>
#include <DiscreteWaveletTransform.hpp>
#include <chrono>

auto time_ev ( const Typedefs::vcpx&, const std::unique_ptr<FourierTransform>&) -> long unsigned int;

template <unsigned long matrix_size>
auto time_ev_dwt (const  Typedefs::vec& x, const DiscreteWaveletTransform<matrix_size> dwt) -> long unsigned int
{  
    Typedefs::vec real_signal = x;
    auto start = std::chrono::high_resolution_clock::now();
    dwt(real_signal, false);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return duration1.count();
}

#endif