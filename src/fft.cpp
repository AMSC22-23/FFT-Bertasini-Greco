#include <fft.hpp>
#include <typedefs.hpp>

using namespace std;

// compute fft
auto recursive::fft(vcpx& x, const bool is_inverse, const int n_cores) -> void
{
    size_t N = x.size();
    // if not power of 2 add zeros
    if (N == 1) return;
    vcpx x_even(N/2, 0);
    vcpx x_odd(N/2, 0);
    for (size_t i = 0; i < N/2; i++){
        x_even[i] = x[2*i];
        x_odd[i] = x[2*i+1];
    }
    fft(x_even, is_inverse, n_cores);
    fft(x_odd, is_inverse, n_cores);
    vcpx X(N, 0);
    for (size_t k = 0; k < N/2; k++){
        cpx W = polar(1.0, (1-2*is_inverse)*-2*M_PI*k/N) * x_odd[k];
        X[k] = x_even[k] + W;
        X[k+N/2] = x_even[k] - W;
    }
    x = X;
}