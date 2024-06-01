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
    cerr << "Output folder: " << output_folder << "\n";
    generate_signal_to_file(output_folder);
    // cerr << "Signal generated\n";
    // fft_tests(output_folder);
    // cerr << "Dwt test starts\n";
    dwt_tests(output_folder);

    return 0;
}