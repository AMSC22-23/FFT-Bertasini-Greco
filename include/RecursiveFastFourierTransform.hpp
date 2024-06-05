/**
 * @file RecursiveFastFourierTransform.hpp
 * @brief Defines the RecursiveFastFourierTransform class for recursive Fast Fourier Transform (FFT) operations.
 */

#ifndef RECURSIVE_FAST_FOURIER_TRANSFORM_HPP
#define RECURSIVE_FAST_FOURIER_TRANSFORM_HPP

#include "FourierTransform.hpp"

namespace tr
{
    /**
     * @class RecursiveFastFourierTransform
     * @brief Represents a recursive Fast Fourier Transform operation.
     */
    class RecursiveFastFourierTransform : public FourierTransform
    {
    private:
        /**
         * @brief Perform the FFT on a complex signal using a recursive approach.
         * @param signal The input signal in the complex domain.
         * @param is_inverse A flag indicating whether to perform the inverse transform.
         */
        auto fft(Typedefs::vcpx &signal, const bool is_inverse) const -> void;

    public:
        /**
         * @brief Default constructor for the RecursiveFastFourierTransform class.
         */
        RecursiveFastFourierTransform() : FourierTransform(){};

        /**
         * @brief Perform the recursive Fast Fourier Transform operation on a complex signal.
         * @param signal The input signal in the complex domain.
         * @param is_inverse A flag indicating whether to perform the inverse transform.
         */
        auto operator()(Typedefs::vcpx &signal, const bool is_inverse) const -> void override;
    };
}

#endif