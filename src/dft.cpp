#include <dft.hpp>
#include <typedefs.hpp>

using namespace std;

auto dft(vcpx& x, bool is_inverse) -> void
{
    unsigned int N = x.size();
    vector<complex<double>> X(N, 0);
    for (unsigned int k = 0; k < N; k++){
        for (unsigned int n = 0; n < N; n++){
            X[k] += x[n] * polar(1.0, (1-2*is_inverse)*-2*M_PI*k*n/N);
        }
    }
    x = X;
}

