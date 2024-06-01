#include <IterativeFastFourierTransform.hpp>
#include <omp.h>
#include <bitreverse.hpp>

#if USE_CUDA == 1
#include "IterativeFastFourierCUDA.cuh"
#endif

using namespace std;
using namespace Typedefs;

#if USE_CUDA == 0
auto IterativeFastFourierTransform::fft (vcpx& x, const bool is_inverse) const -> void 
{
    if (n_cores != -1) omp_set_num_threads(n_cores);
    size_t N = x.size();
    if (N == 1) return;
    // bit reverse copy of x
    bit_reverse_copy(x);
    // butterfly
    for (size_t s = 1; s <= log2(N); s++){
        auto m = (size_t)pow(2, s);
        cpx Wm = polar(1.0, (1-2*is_inverse)*-2*M_PI/m);
        #pragma omp parallel for schedule(static) 
        for (size_t k = 0; k < N; k += m){
            cpx W = 1;
            for (size_t j = 0; j < m/2; j++){
                cpx t = W * x[k+j+m/2];
                cpx u = x[k+j];
                x[k+j] = u + t;
                x[k+j+m/2] = u - t;
                W *= Wm;
            }
        }
    }
}

#else
auto IterativeFastFourierTransform::fft (vcpx& x, const bool is_inverse) const -> void {
  fftCU(x, is_inverse);
}
#endif

auto IterativeFastFourierTransform::operator()(vcpx& x, const bool is_inverse) const -> void
{
    fft(x, is_inverse);
}