#include <iostream>
#include <fstream>

#include <typedefs.hpp>

#include <DiscreteFourierTransform.hpp>
#include <RecursiveFastFourierTransform.hpp>
#include <IterativeFastFourierTransform.hpp>

#include <bitreverse.hpp>
#include <utils.hpp>

#include <time_evaluator.hpp>

using namespace std;
using namespace Typedefs;

void time_evaluation (const vcpx& s, ofstream& output_file)

{
    // unique_ptr<FourierTransform> dft = make_unique<DiscreteFourierTransform>();
    unique_ptr<FourierTransform> i_fft = make_unique<IterativeFastFourierTransform>();
    unique_ptr<FourierTransform> r_fft = make_unique<RecursiveFastFourierTransform>();
    
    auto time_i = time_ev(s, i_fft);
    auto time_r = time_ev(s, r_fft);
    //auto time_d = time_ev(s.get_signal(), dft);
    cout << "-----------------------------------------" << endl;
    cout << "Time for optimal iterative fft: " << time_i << " µs\n";
    cout << "Time for recursive fft: " << time_r << " µs\n";
    //cout << "Time for dft: " << time_d << " µs\n";

    dynamic_cast<IterativeFastFourierTransform*>(i_fft.get())->set_n_cores(1);
    cout << "-----------------------------------------" << endl;
    auto time_0 = time_ev(s, i_fft);
    cout << "Time for  1 processor:  " << time_0 << " µs\n";

    for (int i = 2; i < 20; i++) {
        dynamic_cast<IterativeFastFourierTransform*>(i_fft.get())->set_n_cores(i);
        auto time = time_ev(s, i_fft);
        cout << "Time for " << (i < 10 ? " " : "") << i << " processors: " << time << " µs | ";
        cout << "Speedup: " << (double)time_0 / time << "\n";
        output_file << i << "," << (double)time_0 / time << "\n";
    }
    cout << "-----------------------------------------" << endl;

}

void scaling_evaluation (ofstream& output_file)
{
   // generate incrisingly bigger signals
    vector<double> freqs = {1, 100};
    vector<double> amps = {1, 0.1};

    shared_ptr<FourierTransform> fft_signal = make_shared<IterativeFastFourierTransform>();
    unique_ptr<FourierTransform> fft = make_unique<IterativeFastFourierTransform>();

    // up to 2^24

    auto max_N = (size_t)16777216;
    auto base_signal = vcpx();
    auto x = vector<double>();
    
    x.resize(max_N, 0);
    generate(x.begin(), x.end(), [i = 0, x]() mutable {return i++ * M_PI * 4 / (double)x.size();});
    base_signal.resize(max_N, 0);
    
    for (size_t i = 0; i < freqs.size(); i++)
        for (size_t n = 0; n < max_N; n++)
            base_signal[n] += amps[i] * sin(freqs[i] * x[n]);

    for (size_t N = 2; N <= max_N; N *= 2) {
        auto signal = vcpx(base_signal.begin(), base_signal.begin() + N);
        auto time = time_ev(signal, fft);
        cout << "Time for N = " << N << ": " << time << " µs\n";
        output_file << N << "," << time << "\n";
    }
}

auto fft_tests(const string& output_folder) -> int
{
    // output folder / name
    const string signal_file = output_folder + string("/signal.txt");

    const string transformed_file = output_folder + string("/transformed.txt");
    const string scalability_time = output_folder + string("/scalability_time.txt");
    const string scalability_speedup = output_folder + string("/scalability_speedup.txt");


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

    vec real_signal;
    read_signal(signal_file, real_signal);

    cerr << "Signal read\n";

    // FourierTransform& fft = *make_shared<IterativeFastFourierTransform>();
    IterativeFastFourierTransform fft;


    vcpx signal;

    // fill complex signal
    signal.resize(real_signal.size(), 0);
    transform(real_signal.begin(), real_signal.end(), signal.begin(), [](double d){return cpx(d, 0);});

    cerr << "Complessato\n";

    vcpx transformed_signal = signal;
    
    fft(transformed_signal, false);

    cerr << "Transformed\n";

    auto fft_freqs = vector<double>();

    fft_freqs.resize(transformed_signal.size(), 0);
    transform(
        transformed_signal.begin(), 
        transformed_signal.end(), 
        fft_freqs.begin(), 
        [](cpx c){ return abs(c); }
    );
    fft_freqs.resize(fft_freqs.size() / 2);

    // write signal to file
    // vector<double> real_signal(signal.size(), 0);
    // transform(signal.begin(), signal.end(), real_signal.begin(), [](cpx c){return c.real();});

    // output_file_signal.precision(numeric_limits<double>::max_digits10);
    // for (auto i : real_signal) output_file_signal << i << ",";
    // output_file_signal.seekp(-1, ios_base::end);    
    // output_file_signal << "\n";

    // output should be with full precision
    output_file_transformed.precision(numeric_limits<double>::max_digits10);
    for (auto i : transformed_signal) output_file_transformed << "(" << i.real() << "+" << i.imag() << "j)\n";

    // output_file_signal.close();
    output_file_transformed.close();

    scaling_evaluation(output_file_scalability_time);
    time_evaluation(signal, output_file_scalability_speedup);

    return 0;
}