#include <DiscreteFourierTransform.hpp>
#include <cmath>

using namespace std;
using namespace Typedefs;

auto DiscreteFourierTransform::dft(vcpx& x, const bool is_inverse) const -> void
{
    size_t N = x.size();
    vcpx X(N, 0);
    for (size_t k = 0; k < N; k++){
        for (size_t n = 0; n < N; n++){
            X[k] += x[n] * polar(1.0, (1-2*is_inverse)*-2*M_PI*k*n/N);
        }
    }
    x = X;
}

auto DiscreteFourierTransform::operator()(vcpx& x, const bool is_inverse) const -> void
{
    dft(x, is_inverse);
}