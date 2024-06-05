/**
* @file DiscreteFourierTransform.hpp
* @brief Defines the DiscreteFourierTransform class for discrete Fourier transform operations.
*/

#ifndef DISCRETE_FOURIER_TRANSFORM_HPP
#define DISCRETE_FOURIER_TRANSFORM_HPP

#include "FourierTransform.hpp"

/**
* @class DiscreteFourierTransform
* @brief Represents a discrete Fourier transform operation.
*/
class DiscreteFourierTransform : public FourierTransform {
private:
   /**
    * @brief Perform the discrete Fourier transform on a complex signal.
    * @param signal The input signal in the complex domain.
    * @param is_inverse A flag indicating whether to perform the inverse transform.
    */
   auto dft(Typedefs::vcpx& signal, const bool is_inverse) const -> void;

public:
   /**
    * @brief Default constructor for the DiscreteFourierTransform class.
    */
   DiscreteFourierTransform() : FourierTransform() {};

   /**
    * @brief Perform the discrete Fourier transform operation on a complex signal.
    * @param signal The input signal in the complex domain.
    * @param is_inverse A flag indicating whether to perform the inverse transform.
    */
   auto operator()(Typedefs::vcpx& signal, const bool is_inverse) const -> void override;
};

#endif