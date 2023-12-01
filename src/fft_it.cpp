#include <fft_it.hpp>
#include <bitreverse.hpp>
#include <typedefs.hpp>
#include <cmath>

using namespace std;

auto iterative::fft (vcpx& x) -> void {
    unsigned int N = x.size();
    // if not power of 2 add zeros
    if (N == 1) return;
    // bit reverse copy of x
    bit_reverse_copy(x);
    // butterfly
    for (unsigned int s = 1; s <= log2(N); s++){
        auto m = (unsigned int)pow(2, s);
        cpx Wm = polar(1.0, -2*M_PI/m);
        for (unsigned int k = 0; k < N; k += m){
            cpx W = 1;
            for (unsigned int j = 0; j < m/2; j++){
                cpx t = W * x[k+j+m/2];
                cpx u = x[k+j];
                x[k+j] = u + t;
                x[k+j+m/2] = u - t;
                W *= Wm;
            }
        }
    }
}