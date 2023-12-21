#include <iostream>
#include <fstream>

#include <typedefs.hpp>

#include <Signal.hpp>

#include <DiscreteFourierTransform.hpp>
#include <RecursiveFastFourierTransform.hpp>
#include <IterativeFastFourierTransform.hpp>

#include <time_evaluator.hpp>

using namespace std;
using namespace Typedefs;

void time_evaluation (const Signal& s, ofstream& output_file)

{
    // unique_ptr<FourierTransform> dft = make_unique<DiscreteFourierTransform>();
    unique_ptr<FourierTransform> i_fft = make_unique<IterativeFastFourierTransform>(-1);
    unique_ptr<FourierTransform> r_fft = make_unique<RecursiveFastFourierTransform>();
    
    auto time_i = time_ev(s.get_signal(), i_fft);
    auto time_r = time_ev(s.get_signal(), r_fft);
    //auto time_d = time_ev(s.get_signal(), dft);
    cout << "-----------------------------------------" << endl;
    cout << "Time for optimal iterative fft: " << time_i << " µs\n";
    cout << "Time for recursive fft: " << time_r << " µs\n";
    //cout << "Time for dft: " << time_d << " µs\n";

    i_fft->set_n_cores(1);
    cout << "-----------------------------------------" << endl;
    auto time_0 = time_ev(s.get_signal(), i_fft);
    cout << "Time for  1 processor:  " << time_0 << " µs\n";

    for (int i = 2; i < 20; i++) {
        i_fft->set_n_cores(i);
        auto time = time_ev(s.get_signal(), i_fft);
        cout << "Time for " << (i < 10 ? " " : "") << i << " processors: " << time << " µs | ";
        cout << "Speedup: " << (double)time_0 / time << "\n";
        output_file << i << "," << (double)time_0 / time << "\n";
    }
    cout << "-----------------------------------------" << endl;

}

void scaling_evaluation (ofstream& output_file)
{
   // generate incrisingly bigger signals
    int N = 2;
    vector<double> freqs = {1, 100};
    vector<double> amps = {1, 0.1};

    shared_ptr<FourierTransform> fft_signal = make_shared<IterativeFastFourierTransform>(-1);
    unique_ptr<FourierTransform> fft = make_unique<IterativeFastFourierTransform>(-1);

    // up to 2^24
    for (N = 2; N <= 16777216; N *= 2) {
        Signal s(freqs, amps, N, fft_signal);
        auto time = time_ev(s.get_signal(), fft);
        cout << "Time for N = " << N << ": " << time << " µs\n";
        output_file << N << "," << time << "\n";
    }
}

auto main(int argc, char ** argv) -> int
{
    srand(time(nullptr));

    if (argc != 2) {
        cout << "Usage: " << argv[0] << " output_folder\n";
        return 1;
    }

    // output folder / name
    const string signal_file = argv[1] + string("/signal.txt");
    const string transformed_file = argv[1] + string("/transformed.txt");
    const string scalability_time = argv[1] + string("/scalability_time.txt");
    const string scalability_speedup = argv[1] + string("/scalability_speedup.txt");

    ofstream output_file_signal(signal_file);
    if (!output_file_signal.is_open()) {
        cout << "Could not open file " << signal_file << '\n';
        return 1;
    }

    ofstream output_file_transformed(transformed_file);
    if (!output_file_transformed.is_open()) {
        cout << "Could not open file " << transformed_file << '\n';
        return 1;
    }

    ofstream output_file_scalability_time(scalability_time);
    if (!output_file_scalability_time.is_open()) {
        cout << "Could not open file " << scalability_time << '\n';
        return 1;
    }

    ofstream output_file_scalability_speedup(scalability_speedup);
    if (!output_file_scalability_speedup.is_open()) {
        cout << "Could not open file " << scalability_speedup << '\n';
        return 1;
    }

    // generate signal
    // create N = 2^23
    const int N = 8388608;
    vector<double> freqs = {1};
    vector<double> amps = {1};

    const int number_of_noises = 100;
    // create noised with random freqs (high) and amps (low)
    for (int i = 0; i < number_of_noises; i++) {
        freqs.push_back(arc4random() % 1000 + 200);
        amps.push_back((arc4random() % 100) / 1000.0);
    }

    shared_ptr<FourierTransform> fft = make_shared<IterativeFastFourierTransform>(-1);

    Signal s(freqs, amps, N, fft);

    // write signal to file
    auto signal = s.get_real_signal();
    auto transformed = s.get_transformed_signal();

    output_file_signal.precision(numeric_limits<double>::max_digits10);
    for (auto i : signal) output_file_signal << i << ",";
    output_file_signal.seekp(-1, ios_base::end);    
    output_file_signal << "\n";

    // output should be with full precision
    output_file_transformed.precision(numeric_limits<double>::max_digits10);
    for (auto i : transformed) output_file_transformed << "(" << i.real() << "+" << i.imag() << "j)\n";

    output_file_signal.close();
    output_file_transformed.close();

    scaling_evaluation(output_file_scalability_time);
    time_evaluation(s, output_file_scalability_speedup);

    return 0;
}