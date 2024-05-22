#include <bitreverse.hpp>
#include <typedefs.hpp>
#include <omp.h>

#include <bitset>
#include <cmath>

using namespace Typedefs;

auto bit_reverse(size_t i, size_t N) -> size_t
{
    auto m = static_cast<size_t>(log2(N));
    std::bitset<sizeof(size_t) * 8> b(i);
    std::bitset<sizeof(size_t) * 8> reversed_b;

    for (size_t j = 0; j < m; j++) {
        reversed_b[j] = b[m - j - 1];
    }
    return reversed_b.to_ullong();
}

template <typename T>
auto bit_reverse_copy(T& v) -> void
{
    size_t N = v.size();
    T v_copy = v;
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++){
        v[i] = v_copy[bit_reverse(i, N)];
    }
}

auto next_power_of_2(size_t n) -> size_t
{
    n--;
    n |= n >> 1;   
    n |= n >> 2;   
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;  
    n++;
    return n;
}

template void bit_reverse_copy<vec>(vec& v);
template void bit_reverse_copy<vcpx>(vcpx& v);