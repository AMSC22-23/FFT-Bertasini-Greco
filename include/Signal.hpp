#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <typedefs.hpp>

class Signal {
    private:
    std::vector<double> freqs;
    std::vector<double> amps;
    std::vector<double>x;
    vcpx signal;
    auto generate_signal(unsigned int n_samples) -> void;
    vcpx transformed_signal;
    std::vector<double> fft_freqs;
    auto compute_freqs() -> void;
    auto inverse_transform_signal(const std::function<auto (vcpx&) -> void>& fft) -> void;

    public:
    Signal(std::vector<double> freqs, std::vector<double> amps, unsigned int N, bool padding = true);
    Signal(vcpx signal);
    Signal(const Signal& other) = default;
    Signal(Signal&& other) = default;
    auto operator=(const Signal& other) -> Signal& = default;
    auto operator=(Signal&& other) -> Signal& = default;
    ~Signal() = default;

    auto transform_signal(const std::function<auto (vcpx&) -> void>& fft) -> void;
    auto filter_freqs(const unsigned int flat_freq, const std::function<auto (vcpx&) -> void>& fft) -> void;
    auto get_signal() const -> const vcpx&;
    auto get_real_signal() const -> std::vector<double>;
    auto get_x() const -> const std::vector<double> &;
    auto get_transformed_signal() const -> const vcpx&;
    auto get_fft_freqs() const -> const std::vector<double>&;
};

#endif