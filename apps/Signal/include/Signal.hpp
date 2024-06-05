/**
 * @file Signal.hpp
 * @brief Defines the Signal class for generating and transforming signals.
 */

#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <typedefs.hpp>
#include <FourierTransform.hpp>
#include <memory>

/**
 * @class Signal
 * @brief Class for generating and transforming signals using Fourier Transform.
 */
class Signal {
    private:
    Typedefs::vec freqs; ///< Frequencies of the signal components
    Typedefs::vec amps; ///< Amplitudes of the signal components
    Typedefs::vec x; ///< Sample points of the signal
    Typedefs::vec signal; ///< Generated signal
    std::unique_ptr<Transform<Typedefs::vec>> tr; ///< Transform object for signal processing
    std::unique_ptr<Transform<Typedefs::vec>::InputSpace> input_space; ///< Input space for the transform
    std::unique_ptr<Transform<Typedefs::vec>::OutputSpace> output_space; ///< Output space for the transform

    /**
     * @brief Generates the signal based on frequencies and amplitudes.
     * @param n_samples Number of samples to generate.
     */
    auto generate_signal(size_t n_samples) -> void;

    /**
     * @brief Transforms the signal using the specified transform.
     */
    auto transform_signal() -> void;

    public:
    /**
     * @brief Constructor for the Signal class.
     * @param freqs Vector of frequencies for the signal components.
     * @param amps Vector of amplitudes for the signal components.
     * @param N Number of samples.
     * @param tr Unique pointer to a Transform object for signal processing.
     * @param padding Boolean indicating if padding should be applied (default is true).
     */
    Signal(Typedefs::vec freqs, Typedefs::vec amps, size_t N, std::unique_ptr<Transform<Typedefs::vec>> tr, bool padding = true);

    /**
     * @brief Denoises the signal by filtering out frequencies above the specified flat frequency.
     * @param flat_freq Frequency threshold for filtering.
     */
    auto denoise(const double flat_freq) -> void;

    /**
     * @brief Gets the generated signal.
     * @return Constant reference to the generated signal vector.
     */
    auto get_signal() const -> const Typedefs::vec&;

    /**
     * @brief Gets the sample points of the signal.
     * @return Constant reference to the sample points vector.
     */
    auto get_x() const -> const Typedefs::vec&;

    /**
     * @brief Gets the frequencies of the signal components.
     * @return Vector of frequencies.
     */
    auto get_freqs() const -> const Typedefs::vec;
};

#endif // SIGNAL_HPP
