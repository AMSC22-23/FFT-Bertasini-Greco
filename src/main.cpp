#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

#include <typedefs.hpp>
#include <matplotlibcpp.h>
#include <dft.hpp>
#include <fft.hpp>

namespace plt = matplotlibcpp;
using namespace std;

// generate signal as sum of sin with different frequencies
vcpx generate_signal(vector<double> x, vector<double> freqs, vector<double> amps, int N)
{
    vcpx y(N, 0);
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
    vcpx y = generate_signal(x, freqs, amps, N);
    
    //if not power of 2 add zeros
    if (N != pow(2, floor(log2(N)))) y.resize(pow(2, ceil(log2(N))), 0);

    // compute fft
    auto FFT = fft(y);

    vector<double> FFT_freqs(FFT.size(), 0);
    // filter conjugate symmetric frequencies
    transform(FFT.begin(), FFT.end(), FFT_freqs.begin(), [](cpx c){if (c.imag() < 0) return abs(c); else return 0.0;});

    // filter high frequencies
    for (size_t i = 0; i < FFT.size(); i++) if (i > 50) FFT[i] = 0;
    
    // compute ifft
    vcpx y2 = ifft(FFT);

    auto FFT_filtered = fft(y2);
    vector<double> FFT_filtered_freqs(FFT_filtered.size(), 0);
    // filter conjugate frequencies
    transform(FFT_filtered.begin(), FFT_filtered.end(), FFT_filtered_freqs.begin(), [](cpx c){if (c.imag() < 0) return abs(c); else return 0.0;});

    // compute real of y/y2
    vector<double> y_real(y.size(), 0);
    transform(y.begin(), y.end(), y_real.begin(), [](cpx c){return c.real();});

    vector<double> y2_real(y2.size(), 0);
    transform(y2.begin(), y2.end(), y2_real.begin(), [](cpx c){return c.real();});

    // truncate to N
    y_real.resize(N);
    y2_real.resize(N);

    // plot original, dft, idft
    int rows = 2; int cols = 2;
    plt::figure_size(1400, 700);
    plt::subplot(rows, cols, 1);
    plt::title("Original signal");
    plt::plot(x, y_real);
    plt::subplot(rows, cols, 2);
    plt::title("Inverse FFT");
    plt::plot(x, y2_real);
    plt::subplot(rows, cols, 3);
    plt::title("FFT");
    plt::plot(FFT_freqs);
    plt::subplot(rows, cols, 4);
    plt::title("filtered FFT");
    plt::plot(FFT_filtered_freqs);
    plt::show();
    plt::save("output/fft.png");
}