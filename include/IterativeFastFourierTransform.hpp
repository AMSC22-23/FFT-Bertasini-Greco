/**
 * @file IterativeFastFourierTransform.hpp
 * @brief Defines the IterativeFastFourierTransform class for iterative Fast Fourier Transform (FFT) operations.
 */

#ifndef ITERATIVE_FAST_FOURIER_TRANSFORM_HPP
#define ITERATIVE_FAST_FOURIER_TRANSFORM_HPP

#include "FourierTransform.hpp"

namespace tr
{
    /**
     * @class IterativeFastFourierTransform
     * @brief Represents an iterative Fast Fourier Transform operation.
     */
    class IterativeFastFourierTransform : public FourierTransform
    {
    private:
        /**
         * @brief Perform the FFT on a complex signal.
         * @param signal The input signal in the complex domain.
         * @param is_inverse A flag indicating whether to perform the inverse transform.
         */
        auto fft(Typedefs::vcpx &signal, const bool is_inverse) const -> void;

        int n_cores; /**< The number of cores to be used for parallel computation. */

    public:
        /**
         * @brief Constructor for the IterativeFastFourierTransform class.
         * @param n_cores The number of cores to be used for parallel computation (default: -1, meaning automatic detection).
         */
        IterativeFastFourierTransform(const int n_cores = -1) : FourierTransform(), n_cores(n_cores){};

        /**
         * @brief Set the number of cores to be used for parallel computation.
         * @param n_cores The number of cores.
         */
        auto set_n_cores(const int n_cores) -> void { this->n_cores = n_cores; };

        /**
         * @brief Perform the iterative Fast Fourier Transform operation on a complex signal.
         * @param signal The input signal in the complex domain.
         * @param is_inverse A flag indicating whether to perform the inverse transform.
         */
        auto operator()(Typedefs::vcpx &signal, const bool is_inverse) const -> void override;
    };
}

#endif