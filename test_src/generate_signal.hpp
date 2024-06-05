#include <iostream>
#include <fstream>

#include "typedefs.hpp"

#include "DiscreteFourierTransform.hpp"
#include "RecursiveFastFourierTransform.hpp"
#include "IterativeFastFourierTransform.hpp"

#include "bitreverse.hpp"
#include "utils.hpp"

#include "time_evaluator.hpp"

auto generate_signal_to_file(const std::string& output_folder) -> int
{
    srand(time(nullptr));

    const std::string signal_file = output_folder + std::string("/signal.txt");

    std::ofstream output_file_signal(signal_file);
    if (!output_file_signal.is_open()) {
        std::cout << "Could not open file " << signal_file << '\n';
        return 1;
    }

    const int N = 8388608;
    Typedefs::vec freqs = {1};
    Typedefs::vec amps = {1};
    bool padding = true;

    const int number_of_noises = 100;
    // create noised with random freqs (high) and amps (low)
    for (int i = 0; i < number_of_noises; i++) {
        freqs.push_back(arc4random() % 1000 + 200);
        amps.push_back((arc4random() % 100) / 1000.0);
    }

    auto x = Typedefs::vec();
    auto signal = Typedefs::vec();

    auto is_padding_needed = N & (N - 1);
    auto correct_padding = (is_padding_needed && padding) ? tr::utils::next_power_of_2(N) : N;

    x.resize(N * 2, 0);
    signal.resize(correct_padding, 0);
    
    generate(x.begin(), x.end(), [i = 0, x]() mutable {return i++ * M_PI * 4 / (double)x.size();});
    for (size_t i = 0; i < freqs.size(); i++)
        for (size_t n = 0; n < N; n++)
            signal[n] += amps[i] * sin(freqs[i] * x[n]);

    x.resize(correct_padding);


    //write signal to file
    output_file_signal.precision(std::numeric_limits<double>::max_digits10);
    for (auto i : signal) output_file_signal << i << ",";
    output_file_signal.seekp(-1, std::ios_base::end);    
    output_file_signal << "\n";

    output_file_signal.close();

    return 0;
}
