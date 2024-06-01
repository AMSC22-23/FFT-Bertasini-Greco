#include "generate_signal.hpp"
#include "test_fft.hpp"
#include "test_dwt.hpp"

#include <iostream>

using namespace std;

int main ( int argc, char* argv[] )
{
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <output_folder>" << endl;
        return 1;
    }
    string output_folder = argv[1];

    cout << endl;
    cout <<"-----------------------------------------\n";
    cout << "TRANSFORMS TESTS START" << endl;
    cout <<"-----------------------------------------\n";


    //generate signal up to 2^24
    generate_signal_to_file(output_folder);
    cerr << "\nSignal generated.\n";

    // FFT tests
    cout << endl;
    cout <<"-----------------------------------------\n";
    cerr << "FFT test starts\n";
    cout <<"-----------------------------------------\n";
    fft_tests(output_folder);

    // DWT tests
    cout << endl;
    cout <<"-----------------------------------------\n";
    cerr << "DWT test starts\n";
    cout <<"-----------------------------------------\n";
    dwt_tests(output_folder);

    return 0;
}