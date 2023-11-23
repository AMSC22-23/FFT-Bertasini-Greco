#include <dft.hpp>
#include <typedefs.hpp>

using namespace std;

vcpx dft(vcpx x)
{
    int N = x.size();
    vector<complex<double>> X(N, 0);
    for (int k = 0; k < N; k++){
        for (int n = 0; n < N; n++){
            X[k] += x[n] * polar(1.0, -2*M_PI*k*n/N);
        }
        X[k] /= N;
    }
    return X;
}

vcpx idft(vcpx X)
{   
    int N = X.size();
    vcpx x(N, 0);
    for (int n = 0; n < N; n++){
        for (int k = 0; k < N; k++){
            x[n] += X[k] * polar(1.0, 2*M_PI*k*n/N);
        }
        x[n] /= N;
    }
    return x;
}