#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <typedefs.hpp>
#include <fft_it.hpp>

class Signal {
    private:
    std::vector<double> freqs;
    std::vector<double> amps;
    std::vector<double>x;
    vcpx signal;
    auto generate_signal(unsigned int n_samples) -> void;
    vcpx transformed_signal;
    std::vector<double> fft_freqs;
    std::function<auto (vcpx&) -> void> fft;
    auto compute_freqs() -> void;
    auto transform_signal() -> void;
    auto inverse_transform_signal() -> void;

    public:
    Signal(std::vector<double> freqs, std::vector<double> amps, unsigned int N, const std::function<auto (vcpx&) -> void>& _fft = iterative::fft, bool padding = true);
    Signal(const Signal& other) = default;
    Signal(Signal&& other) = default;
    auto operator=(const Signal& other) -> Signal& = default;
    auto operator=(Signal&& other) -> Signal& = default;
    ~Signal() = default;

    auto filter_freqs(const unsigned int flat_freq) -> void;
    auto get_signal() const -> const vcpx&;
    auto get_real_signal() const -> std::vector<double>;
    auto get_x() const -> const std::vector<double> &;
    auto get_transformed_signal() const -> const vcpx&;
    auto get_fft_freqs() const -> const std::vector<double>&;
};

#endif