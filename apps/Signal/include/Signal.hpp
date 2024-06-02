#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <typedefs.hpp>
#include <FourierTransform.hpp>
#include <memory>

class Signal {
    private:
    Typedefs::vec freqs;
    Typedefs::vec amps;
    Typedefs::vec x;
    Typedefs::vec signal;
    auto generate_signal(size_t n_samples) -> void;
    std::unique_ptr<Transform<Typedefs::vec>> tr;
    std::unique_ptr<Transform<Typedefs::vec>::InputSpace> input_space;
    std::unique_ptr<Transform<Typedefs::vec>::OutputSpace> output_space;

    auto transform_signal() -> void;

    public:
    Signal(Typedefs::vec freqs, Typedefs::vec amps, size_t N, std::unique_ptr<Transform<Typedefs::vec>> tr, bool padding = true);

    auto denoise(const double flat_freq) -> void;
    auto get_signal() const -> const Typedefs::vec&;
    auto get_x()      const -> const Typedefs::vec&;
    auto get_freqs() const -> const Typedefs::vec;
};

#endif