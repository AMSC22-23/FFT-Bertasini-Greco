#include <bitreverse.hpp>
#include <typedefs.hpp>
#include <omp.h>

auto bit_reverse(unsigned int i, unsigned int N) -> unsigned int
{
    auto m = (unsigned int)log2(N);
    unsigned int b = i;
    b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
    b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
    b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
    b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
    b = ((b >> 16) | (b << 16)) >> (32 - m);
    return b;
}

auto bit_reverse_copy(vcpx& v) -> void
{
    unsigned int N = v.size();
    vcpx v_copy = v;
    #pragma omp parallel for
    for (unsigned int i = 0; i < N; i++){
        v[i] = v_copy[bit_reverse(i, N)];
    }
}

auto next_power_of_2(long unsigned int n) -> long unsigned int
{
    n--;
    n |= n >> 1;   // Divide by 2^k for consecutive doublings of k up to 32,
    n |= n >> 2;   // and then or the results.
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;  // This is now a number of the form 2^k - 1 (where
    n++;
    return n;
}