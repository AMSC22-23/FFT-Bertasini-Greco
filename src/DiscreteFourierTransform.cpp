#include <cmath>

#include "DiscreteFourierTransform.hpp"

using namespace std;
using namespace Typedefs;

auto DiscreteFourierTransform::dft(vcpx& x, const bool is_inverse) const -> void
{
    const size_t N = x.size();
    vcpx X(N, 0);
    const double mult = (1-2*is_inverse)*-2*M_PI/N;
    for (size_t k = 0; k < N; k++){
        for (size_t n = 0; n < N; n++){
            X[k] += x[n] * polar(1.0, mult*k*n);
        }
    }
    move(X.begin(), X.end(), x.begin()); 
}

auto DiscreteFourierTransform::operator()(vcpx& x, const bool is_inverse) const -> void
{
    dft(x, is_inverse);
}