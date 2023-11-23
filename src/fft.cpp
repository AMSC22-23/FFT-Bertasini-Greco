#include <fft.hpp>
#include <typedefs.hpp>

using namespace std;

// compute fft
vcpx fft(vcpx x)
{
    int N = x.size();
    // if not power of 2 add zeros
    if (N == 1) return vcpx {x[0]};
    vcpx x_even(N/2, 0);
    vcpx x_odd(N/2, 0);
    for (int i = 0; i < N/2; i++){
        x_even[i] = x[2*i];
        x_odd[i] = x[2*i+1];
    }
    vcpx X_even = fft(x_even);
    vcpx X_odd  = fft(x_odd);
    vcpx X(N, 0);
    for (int k = 0; k < N/2; k++){
        cpx W = polar(1.0, -2*M_PI*k/N) * X_odd[k];
        X[k] = X_even[k] + W;
        X[k+N/2] = X_even[k] - W;
    }
    return X;
}

// compute ifft
vcpx ifft(vcpx X)
{
    transform(X.begin(), X.end(), X.begin(), [ ](cpx c){return conj(c);});
    vcpx x = fft(X);
    transform(x.begin(), x.end(), x.begin(), [ ](cpx c){return conj(c);});
    
    transform(x.begin(), x.end(), x.begin(), [x](cpx c){return cpx(c.real()/x.size(), c.imag()/x.size());});

    return x;
}

