#include "matplotlibcpp.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

namespace plt = matplotlibcpp;
using namespace std;

// compute dft
vector<complex<double>> dft(vector<double> x)
{
    int N = x.size();
    vector<complex<double>> X(N, 0);
    for (int k = 0; k < N; k++){
        for (int n = 0; n < N; n++){
            X[k] += complex<double>(x[n] * cos(2 * M_PI * k * n / N), -x[n] * sin(2 * M_PI * k * n / N));
        }
        X[k] /= N;
    }
    return X;
}

// compute idft
vector<double> idft(vector<complex<double>> X)
{   
    int N = X.size();
    vector<double> x(N, 0);
    for (int n = 0; n < N; n++){
        for (int k = 0; k < N; k++){
            x[n] += X[k].real() * cos(2 * M_PI * k * n / N) - X[k].imag() * sin(2 * M_PI * k * n / N);
        }
        x[n] /= N;
    }
    return x;
}

// generate signal as sum of sin with different frequencies
vector<double> generate_signal(vector<double> x, vector<double> freqs, vector<double> amps, int N)
{
    vector<double> y(N, 0);
    for (size_t i = 0; i < freqs.size(); i++)
        for (int n = 0; n < N; n++)
            y[n] += amps[i] * sin(freqs[i] * x[n]);
    return y;
}

int main()
{
    // generate signal
    int N = 10000;
    vector<double> x (N, 0);
    for (int i = 0; i < N; i++) x[i] = i*M_PI*2/N;

    vector<double> freqs = {1,100};
    vector<double> amps = {1,.1};
    vector<double> y = generate_signal(x, freqs, amps, N);

    // compute dft  
    auto DFT = dft(y);

    vector<double> DFT_freqs(N, 0);
    transform(DFT.begin(), DFT.end(), DFT_freqs.begin(), [](complex<double> c){if (c.imag() < 0) return abs(c); else return 0.0;});

    // remove high frequencies
    for (size_t i = 0; i < DFT_freqs.size(); i++) if (i > 50) DFT[i] = 0;
        
    vector<double> DFT_freqs_filtered(N, 0);
    transform(DFT.begin(), DFT.end(), DFT_freqs_filtered.begin(), [](complex<double> c){return abs(c);});

    // compute idft
    vector<double> y2 = idft(DFT);
    
    // plot original, dft, idft
    plt::figure_size(1200, 700);
    plt::subplot(2, 2, 1);
    plt::title("Original signal");
    plt::plot(x, y);
    plt::subplot(2, 2, 2);
    plt::title("Filtered signal");
    plt::plot(x, y2);
    plt::subplot(2, 2, 3);
    plt::title("DFT");
    plt::plot(DFT_freqs);
    plt::subplot(2, 2, 4);
    plt::title("DFT filtered");
    plt::plot(DFT_freqs_filtered);
    plt::show();
    plt::save("output/dft.png");
}