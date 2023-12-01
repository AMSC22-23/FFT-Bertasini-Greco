#include <ifft.hpp>
#include <typedefs.hpp>

using namespace std;

// compute ifft
auto ifft(vcpx X, const function<auto (vcpx&) -> void>& fft) -> vcpx
{
    transform(X.begin(), X.end(), X.begin(), [ ](cpx c){return conj(c);});
    fft(X);
    transform(X.begin(), X.end(), X.begin(), [ ](cpx c){return conj(c);});
    transform(X.begin(), X.end(), X.begin(), [X](cpx c){return cpx(c.real()/(double)X.size(), c.imag()/(double)X.size());});
    return X;
}
