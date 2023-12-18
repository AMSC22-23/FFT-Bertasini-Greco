#include <dft.hpp>
#include <typedefs.hpp>

using namespace std;

auto dft(vcpx& x, const bool is_inverse, const int) -> void
{
    size_t N = x.size();
    vector<complex<double>> X(N, 0);
    for (size_t k = 0; k < N; k++){
        for (size_t n = 0; n < N; n++){
            X[k] += x[n] * polar(1.0, (1-2*is_inverse)*-2*M_PI*k*n/N);
        }
    }
    x = X;
}

