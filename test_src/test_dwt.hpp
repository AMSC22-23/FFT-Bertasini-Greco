#include <iostream>
#include <fstream>

#include <typedefs.hpp>

#include <DiscreteWaveletTransform.hpp>

#include <bitreverse.hpp>
#include <utils.hpp>

#include <time_evaluator.hpp>


using namespace std;
using namespace Typedefs;

auto dwt_tests(const string& output_folder) -> int
{

    const string signal_file = output_folder + string("/signal.txt");
    const string weak_scalability_test = output_folder + string("/weak_scalability_test_dwt.txt");
    const string strong_scalability_test = output_folder + string("/strong_scalability_test_dwt.txt");

    ofstream output_file_strong(strong_scalability_test);
    if (!output_file_strong.is_open()) {
        cout << "Could not open file " << strong_scalability_test << '\n';
        return 1;
    }

    ofstream output_file_weak(weak_scalability_test);
    if (!output_file_weak.is_open()) {
        cout << "Could not open file " << weak_scalability_test << '\n';
        return 1;
    }


    //read signal from file
    vec real_signal;
    read_signal(signal_file, real_signal);


    //test1: strong scalability
    cout << endl;
    cout << "1. STRONG SCALABILITY TEST:\n";
    DiscreteWaveletTransform dwt(TRANSFORM_MATRICES::DAUBECHIES_D40, 10);
    dwt.set_n_cores(1);
    auto time_0 = time_ev_dwt(real_signal, dwt);
    cout << "Time for   1 processor: " << time_0 << " µs\n";
    output_file_strong << 1 << "," << (double)time_0 << "\n";
    for (int i = 2; i < 20; i++) {
       dwt.set_n_cores(i);
        auto time = time_ev_dwt(real_signal, dwt);
        cout << "Time with " << (i < 10 ? " " : "") << i << " processors: " << time << " µs | ";
        cout << "Speedup: " << (double)time_0 / time << "\n";
        output_file_strong << i << "," << time << "\n";
    }


    //test2: weak scalability
    cout << endl;
    cout << "2. WEAK SCALABILITY TEST:\n";
    int n=pow(2, 18);
    for (int i=1; i<9; i*=2){
        dwt.set_n_cores(i);
        vec signal = vec(real_signal.begin(), real_signal.begin() + n);
        auto elapsed = time_ev_dwt(signal, dwt);
        if (i==1) cout << "Time with " << i << " core: " << elapsed << "  µs "<< "for a signal of size " << signal.size() << "\n";
        else cout << "Time with " << i << " cores: " << elapsed << " µs "<< "for a signal of size " << signal.size() << "\n";
        output_file_weak << i << "," <<signal.size() << "," << elapsed << "\n";
        n*=2;
    }

    // //test3: scalability levels
    // for (int i=1; i<9; i*=2){
    //     DiscreteWaveletTransform dwt_weak(TRANSFORM_MATRICES::DAUBECHIES_D40, 4*i, i);
    //     auto elapsed = time_ev_dwt(real_signal, dwt_weak);
    //     output_file_weak << i << "," << elapsed << "\n";
    //     cout << "Time elapsed for DWT with " << i << " cores/levels: " << elapsed << " microseconds\n";
    // }

    return 0;
}

