#include "utils.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace Typedefs;


auto next_power_of_2(size_t n) -> size_t
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

auto countSubdivisions(int i, int j, int size, int subdivisions) -> int {
    int currentSize = size;

    for (int level = 0; level < subdivisions; ++level) {
        int halfSize = currentSize / 2;

        if (i < halfSize && j < halfSize) {
            // Point is in the top-left submatrix
            currentSize = halfSize;
        } else {
            // Adjust i and j for the next level of subdivision
            if (i >= halfSize) i -= halfSize;
            if (j >= halfSize) j -= halfSize;
            return subdivisions - level - 1; // Subtract 1 to make the count 0-based
        }
    }

    // If the point is not in the top-left submatrix after all subdivisions
    return 0;
} 

auto read_signal(const std::string& signal_file, vec& real_signal) -> void {
  // read signal from file
    ifstream input_file_signal(signal_file);
    if (!input_file_signal.is_open()) {
        cout << "Could not open file " << signal_file << '\n';
        return;
    }

    string line;

    if (getline(input_file_signal, line)) {
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            if (!value.empty()) { 
                try {
                    real_signal.push_back(std::stod(value));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid value found: " << value << std::endl;
                    return;
                }
            }
        }
    }
    input_file_signal.close();
}