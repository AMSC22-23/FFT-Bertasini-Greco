#include <fstream>
#include <iostream>

#include "utils.hpp"

using namespace std;
using namespace Typedefs;
using namespace tr::utils;

auto tr::utils::next_power_of_2(size_t n) -> size_t
{
    n--;
    n |= n >> 1;   
    n |= n >> 2;   
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;  
    n++;
    return n;
}

auto tr::utils::countSubdivisions(unsigned int i, unsigned int j, const unsigned int rows, const unsigned int cols, const unsigned int subdivisions) -> int {
    auto currentRows = rows;
    auto currentCols = cols;

    for (unsigned int level = 0; level < subdivisions; ++level) {
        auto halfRows = currentRows / 2;
        auto halfCols = currentCols / 2;

        if (i < halfRows && j < halfCols) {
            // Point is in the top-left submatrix
            currentRows = halfRows;
            currentCols = halfCols;
        } else {
            // Adjust i and j for the next level of subdivision
            if (i >= halfRows) i -= halfRows;
            if (j >= halfCols) j -= halfCols;
            return subdivisions - level - 1; // Subtract 1 to make the count 0-based
        }
    }

    // If the point is not in the top-left submatrix after all subdivisions
    return 0;
}

auto tr::utils::read_signal(const string& signal_file, vec& real_signal) -> void {
  // read signal from file
    ifstream input_file_signal(signal_file);
    if (!input_file_signal.is_open()) {
        cout << "Could not open file " << signal_file << '\n';
        return;
    }

    string line;

    if (getline(input_file_signal, line)) {
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            if (!value.empty()) { 
                try {
                    real_signal.push_back(stod(value));
                } catch (const invalid_argument& e) {
                    cerr << "Invalid value found: " << value << endl;
                    return;
                }
            }
        }
    }
    input_file_signal.close();
}