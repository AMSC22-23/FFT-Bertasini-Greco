#include <fft_it.hpp>
#include <bitreverse.hpp>
#include <typedefs.hpp>
#include <cmath>
#include <omp.h>

using namespace std;

auto iterative::fft (vcpx& x, const bool is_inverse, const int n_cores) -> void {

    if (n_cores != -1) omp_set_num_threads(n_cores);
    unsigned int N = x.size();
    if (N == 1) return;
    // bit reverse copy of x
    bit_reverse_copy(x);
    // butterfly
    for (unsigned int s = 1; s <= log2(N); s++){
        auto m = (unsigned int)pow(2, s);
        cpx Wm = polar(1.0, (1-2*is_inverse)*-2*M_PI/m);
        #pragma omp parallel for schedule(static) 
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