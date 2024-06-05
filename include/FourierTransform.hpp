/**
* @file FourierTransform.hpp
* @brief Defines the FourierTransform class, which inherits from the Transform class for Fourier transform operations.
*/

#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include "Transform.hpp"

/**
* @class FourierTransform
* @brief Represents a Fourier transform operation.
* @tparam Typedefs::vec The vector type used for input and output data.
*/
class FourierTransform : public Transform<Typedefs::vec> {
protected:
   /**
    * @class InputSpace
    * @brief Represents the input space for the Fourier transform.
    */
   class InputSpace : public Transform::InputSpace {
   public:
       Typedefs::vcpx data; /**< The input data in the complex domain. */

       /**
        * @brief Constructor for the InputSpace class.
        * @param _signal The input signal.
        */
       InputSpace(const Typedefs::vec& _signal);

       /**
        * @brief Get the input data.
        * @return The input data as a vector.
        */
       auto get_data() const -> Typedefs::vec override;
   };

   /**
    * @class OutputSpace
    * @brief Represents the output space for the Fourier transform.
    */
   class OutputSpace : public Transform::OutputSpace {
   public:
       Typedefs::vcpx data; /**< The output data in the complex domain. */

       /**
        * @brief Get the plottable representation of the output data.
        * @return The plottable representation as a vector.
        */
       auto get_plottable_representation() const -> Typedefs::vec override;

       /**
        * @brief Compress the output data using a specified method and compression value.
        * @param method The compression method.
        * @param kept The compression value indicating the amount of data to keep.
        */
       auto compress(const std::string& method, const double kept) -> void override;

       /**
        * @brief Filter frequencies in the output data based on a percentile cutoff.
        * @param percentile_cutoff The percentile cutoff value for frequency filtering.
        */
       auto filter_freqs(const double percentile_cutoff) -> void;

       /**
        * @brief Filter the magnitude of the output data based on a percentile cutoff.
        * @param percentile_cutoff The percentile cutoff value for magnitude filtering.
        */
       auto filter_magnitude(const double percentile_cutoff) -> void;

       /**
        * @brief Denoise the output data by applying a frequency cutoff.
        * @param freq_cutoff The frequency cutoff value for denoising.
        */
       auto denoise(const double freq_cutoff) -> void;
   };

public:
   /**
    * @brief Get the input space for the Fourier transform.
    * @param v The input data.
    * @return A unique pointer to the InputSpace object.
    */
   auto get_input_space(const Typedefs::vec& v) const -> std::unique_ptr<Transform::InputSpace> override;

   /**
    * @brief Get the output space for the Fourier transform.
    * @return A unique pointer to the OutputSpace object.
    */
   auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> override;

   /**
    * @brief Perform the Fourier transform operation.
    * @param in The input space for the transformation.
    * @param out The output space for the transformation.
    * @param is_inverse A flag indicating whether to perform the inverse transform.
    */
   auto operator()(Transform::InputSpace& in, Transform::OutputSpace& out, const bool is_inverse) const -> void override;

   /**
    * @brief Perform the Fourier transform operation on a complex signal.
    * @param signal The input signal in the complex domain.
    * @param is_inverse A flag indicating whether to perform the inverse transform.
    */
   virtual auto operator()(Typedefs::vcpx& signal, const bool is_inverse) const -> void = 0;
};

#endif