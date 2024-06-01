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

// Time comparison between iterative and recursive fft
void time_evaluation (const vcpx& s)
{
    // unique_ptr<FourierTransform> dft = make_unique<DiscreteFourierTransform>();
    unique_ptr<FourierTransform> i_fft = make_unique<IterativeFastFourierTransform>();
    unique_ptr<FourierTransform> r_fft = make_unique<RecursiveFastFourierTransform>();
    
    auto time_i = time_ev(s, i_fft);
    auto time_r = time_ev(s, r_fft);
    //auto time_d = time_ev(s.get_signal(), dft);
    cout << "Time for optimal iterative fft: " << time_i << " µs\n";
    cout << "Time for recursive fft: " << time_r << " µs\n";
    //cout << "Time for dft: " << time_d << " µs\n";
}

//strong scalability
void strong_scaling_evaluation (const vcpx& s, ofstream& output_file)
{
    unique_ptr<FourierTransform> i_fft = make_unique<IterativeFastFourierTransform>();
    dynamic_cast<IterativeFastFourierTransform*>(i_fft.get())->set_n_cores(1);
    auto time_0 = time_ev(s, i_fft);
    cout << "Time for   1 processor:  " << time_0 << " µs\n";
    output_file << 1 << "," << (double)time_0 << "\n";
    for (int i = 2; i < 20; i++) {
        dynamic_cast<IterativeFastFourierTransform*>(i_fft.get())->set_n_cores(i);
        auto time = time_ev(s, i_fft);
        cout << "Time with " << (i < 10 ? " " : "") << i << " processors: " << time << " µs | ";
        cout << "Speedup: " << (double)time_0 / time << "\n";
        output_file << i << "," << time << "\n";
    }
}


// weak scalability
void weak_scaling_evaluation (const vcpx& s, ofstream& output_file)
{
    unique_ptr<FourierTransform> fft = make_unique<IterativeFastFourierTransform>();
    int n=pow(2, 20);
    for (int i=1; i<9; i*=2){
        dynamic_cast<IterativeFastFourierTransform*>(fft.get())->set_n_cores(i);
        vcpx signal = vcpx(s.begin(), s.begin() + n);
        auto elapsed = time_ev(signal, fft);
        cout << "Time with " << i << " cores: " << elapsed << " µs "<< "for a signal of size " << signal.size() << "\n";
        output_file << i << "," <<signal.size() << "," << elapsed << "\n";
        n*=2;
    }
}


auto fft_tests(const string& output_folder) -> int
{
    // output folder / name
    const string signal_file = output_folder + string("/signal.txt");

    const string transformed_file = output_folder + string("/comparison_fft.txt");
    const string weak_scalability = output_folder + string("/weak_scalability_test_fft.txt");
    const string strong_scalability = output_folder + string("/strong_scalability_test_fft.txt");


    ofstream output_file_transformed(transformed_file);
    if (!output_file_transformed.is_open()) {
        cout << "Could not open file " << transformed_file << '\n';
        return 1;
    }

    ofstream output_file_weak_scalability(weak_scalability);
    if (!output_file_weak_scalability.is_open()) {
        cout << "Could not open file " << weak_scalability << '\n';
        return 1;
    }

    ofstream output_file_strong_scalability(strong_scalability);
    if (!output_file_strong_scalability.is_open()) {
        cout << "Could not open file " << strong_scalability << '\n';
        return 1;
    }

    vec real_signal;
    read_signal(signal_file, real_signal);

    IterativeFastFourierTransform fft;
    vcpx signal;

    // fill complex signal
    signal.resize(real_signal.size(), 0);
    transform(real_signal.begin(), real_signal.end(), signal.begin(), [](double d){return cpx(d, 0);});

    vcpx transformed_signal = signal;

    fft(transformed_signal, false);

    auto fft_freqs = vector<double>();

    fft_freqs.resize(transformed_signal.size(), 0);
    transform(
        transformed_signal.begin(), 
        transformed_signal.end(), 
        fft_freqs.begin(), 
        [](cpx c){ return abs(c); }
    );
    fft_freqs.resize(fft_freqs.size() / 2);

    // output should be with full precision
    output_file_transformed.precision(numeric_limits<double>::max_digits10);
    for (auto i : transformed_signal) output_file_transformed << "(" << i.real() << "+" << i.imag() << "j)\n";

    // output_file_signal.close();
    output_file_transformed.close();

    cout << endl;
    cout <<"1. COMPARISON BETWEEN ITERATIVE AND RECURSIVE:" << endl;
    time_evaluation(signal);

    cout << endl;
    cout << "2. STRONG SCALABILITY TEST:\n";
    strong_scaling_evaluation(signal, output_file_strong_scalability);

    cout << endl;
    cout << "3. WEAK SCALABILITY TEST:\n";
    weak_scaling_evaluation(signal, output_file_weak_scalability);

    return 0;
}