#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

#include <typedefs.hpp>
#include <matplotlibcpp.h>

#include <Signal.hpp>

#include <FourierTransform.hpp>
#include <DiscreteFourierTransform.hpp>
#include <RecursiveFastFourierTransform.hpp>
#include <IterativeFastFourierTransform.hpp>
#include <DiscreteWaveletTransform.hpp>

namespace plt = matplotlibcpp;
using namespace std;
using namespace Typedefs;

auto plot_stuff (const Signal& s, const Signal& s_filtered, const int& N, bool truncate = true) {
    const int width = 1400;
    const int height = 700;

    auto x = s.get_x();
    auto y = s.get_signal();
    auto y2 = s_filtered.get_signal();
    
    // truncate to N
    if (truncate) {
        x.resize(N);
        y.resize(N);
        y2.resize(N);
    }
    
    string color = "tan";
    // print x and y sizes 
    int rows = 2; int cols = 2;
    plt::figure_size(width, height);
    plt::subplot(rows, cols, 1);
    plt::title("Original signal");
    plt::plot(x, y, color); // Set plot color to brown
    plt::subplot(rows, cols, 2);
    plt::title("Inverse FFT");
    plt::plot(x, y2, color); // Set plot color to brown
    plt::subplot(rows, cols, 3);
    plt::title("FFT");
    plt::plot(s.get_fft_freqs(), color); // Set plot color to brown
    plt::subplot(rows, cols, 4);
    plt::title("filtered FFT");
    plt::plot(s_filtered.get_fft_freqs(), color); // Set plot color to brown
    plt::save("output/fft.png");
    plt::show();
}

auto main() -> int
{
    srand(time(nullptr));

    // generate signal
    const int N = 8192;
    
    vector<double> freqs = {1, 500};
    vector<double> amps = {1, 0.1};

    shared_ptr<Transform<vec>>fft = make_shared<DiscreteWaveletTransform>(TRANSFORM_MATRICES::HAAR, 4);
    // shared_ptr<Transform<vec>>fft = make_shared<IterativeFastFourierTransform>();

    Signal s(freqs, amps, N, fft);

    Signal s_filtered(freqs, amps, N, fft);

    // const double freq_flat = 50.0;
    const double freq_flat = 0.0625;
    s_filtered.denoise(freq_flat);

    plot_stuff(s, s_filtered, N);

    return 0;
}