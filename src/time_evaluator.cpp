#include <iostream>

#include "time_evaluator.hpp"

using namespace std;
using namespace Typedefs;

 auto time_ev (const vcpx& x, const unique_ptr<FourierTransform>& f) -> long unsigned int
 {
    vcpx x_time = x;
    auto start = chrono::high_resolution_clock::now();
    f->operator()(x_time, false);
    auto stop = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(stop - start);
    return duration1.count();
 }

auto time_ev_dwt (const Typedefs::vec& x, const DiscreteWaveletTransform& dwt) -> long unsigned int
{
    Typedefs::vec real_signal = x;
    auto start = std::chrono::high_resolution_clock::now();
    dwt(real_signal, false);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return duration1.count();
}
