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
using namespace tr;

auto plot_stuff (vec& x, vec& y, vec& y2, vec& freq, vec& freq_filtered, const int N, const bool truncate = false) -> void
{
    const int width = 1400;
    const int height = 700;

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
    plt::title("Inverse");
    plt::plot(x, y2, color); // Set plot color to brown
    plt::subplot(rows, cols, 3);
    plt::title("Transformed");
    plt::plot(freq, color); // Set plot color to brown
    plt::subplot(rows, cols, 4);
    plt::title("Filtered");
    plt::plot(freq_filtered, color); // Set plot color to brown
    plt::save("output/tr.png");
    plt::show();
}

auto main() -> int
{
    srand(time(nullptr));

    // generate signal
    const int N = 8192;
    
    vec freqs = {1, 100};
    vec amps = {1, 0.1};

    cout << "Choose a transform:[newline for default]\n";
    cout << "1. Discrete Wavelet Transform\n";
    cout << "2. Discrete Fourier Transform\n";
    cout << "3. Recursive Fast Fourier Transform\n";
    cout << "4. Iterative Fast Fourier Transform [DEFAULT]\n";
    int choice = 4;

    string tmp;
    getline(cin, tmp);
    if (tmp.empty()) cout << "Using default FFT\n";
    else choice = stoi(tmp);

    unique_ptr<Transform<vec>>tr;
    double freq_flat = 50.0;

    if (choice == 1) {
        freq_flat = 0.00625;
        tr = make_unique<DiscreteWaveletTransform>(TRANSFORM_MATRICES::DAUBECHIES_D40, 10);
    } else if (choice == 2) {
        tr = make_unique<DiscreteFourierTransform>();
    } else if (choice == 3) {
        tr = make_unique<RecursiveFastFourierTransform>();
    } else if (choice == 4) {
        tr = make_unique<IterativeFastFourierTransform>();
    } else {
        cerr << "Invalid choice\n";
        return -1;
    }

    Signal s(freqs, amps, N, std::move(tr));

    vec x = s.get_x();
    vec y = s.get_signal();
    vec freq = s.get_freqs();
  
    s.denoise(freq_flat);

    vec y2 = s.get_signal();
    vec freq_filtered = s.get_freqs();

    plot_stuff(x, y, y2, freq, freq_filtered, N, true);

    return 0;
}