#include <iostream>
#include <fstream>

#include <typedefs.hpp>

#include <Signal.hpp>
#include <dft.hpp>
#include <fft.hpp>
#include <fft_it.hpp>
#include <time_evaluator.hpp>

using namespace std;

void time_evaluation (const Signal& s)
{
    auto time_i = time_ev(s.get_signal(), iterative::fft, -1);
    auto time_r = time_ev(s.get_signal(), recursive::fft, -1);
    //auto time_d = time_ev(s.get_signal(), dft, -1);
    cout << "-----------------------------------------" << endl;
    cout << "Time for optimal iterative fft: " << time_i << " µs\n";
    cout << "Time for recursive fft: " << time_r << " µs\n";
    //cout << "Time for dft: " << time_d << " µs\n";

    cout << "-----------------------------------------" << endl;
    auto time_0 = time_ev(s.get_signal(), iterative::fft, 1);
    cout << "Time for  1 processor:  " << time_0 << " µs\n";

    for (int i = 2; i < 20; i++) {
        auto time = time_ev(s.get_signal(), iterative::fft, i);
        cout << "Time for " << (i < 10 ? " " : "") << i << " processors: " << time << " µs | ";
        cout << "Speedup: " << (double)time_0 / time << "\n";
    }
    cout << "-----------------------------------------" << endl;

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

    // generate signal
    const int N = 1000000;
    vector<double> freqs = {1};
    vector<double> amps = {1};

    const int number_of_noises = 100;
    // create noised with random freqs (high) and amps (low)
    for (int i = 0; i < number_of_noises; i++) {
        freqs.push_back(arc4random() % 1000 + 200);
        amps.push_back((arc4random() % 100) / 1000.0);
    }

    Signal s(freqs, amps, N, iterative::fft);

    // write signal to file
    auto signal = s.get_real_signal();
    auto transformed = s.get_transformed_signal();

    
    for (auto i : signal) output_file_signal << i << ",";
    output_file_signal.seekp(-1, ios_base::end);    
    output_file_signal << "\n";

    // output should be with full precision
    output_file_transformed.precision(numeric_limits<double>::max_digits10);
    for (auto i : transformed) output_file_transformed << "(" << i.real() << "+" << i.imag() << "j)\n";

    output_file_signal.close();
    output_file_transformed.close();

    time_evaluation(s);
    
    return 0;
}