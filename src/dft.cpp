#include <dft.hpp>
#include <typedefs.hpp>

//@note Normally you use 
// #include "name" instead of #include <name> for local includes.
// This is because the compiler will look for the file in the local directory
// before looking in the system directories.
using namespace std;

auto dft(vcpx& x, const bool is_inverse, const int) -> void
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

