/**
 * @file DiscreteWaveletTransform.hpp
 * @brief Defines the DiscreteWaveletTransform class, which inherits from the Transform class for discrete wavelet transform operations.
 */

#ifndef DISCRETE_WAVELET_TRANSFORM_HPP
#define DISCRETE_WAVELET_TRANSFORM_HPP

#include <span>

#include "Transform.hpp"
#include "TransformMatrices.hpp"

namespace tr
{

    /**
     * @class DiscreteWaveletTransform
     * @brief Represents a discrete wavelet transform operation.
     * @tparam Typedefs::vec The vector type used for input and output data.
     */
    class DiscreteWaveletTransform : public Transform<Typedefs::vec>
    {
    private:
        std::span<const Typedefs::DType> transform_matrix; /**< The transform matrix for the forward transform. */
        std::span<const Typedefs::DType> inverse_matrix;   /**< The transform matrix for the inverse transform. */
        uint8_t user_levels = 0;                           /**< The number of levels to be used in the transform. */
        int n_cores;                                       /**< The number of cores to be used for parallel computation. */

    protected:
        /**
         * @class InputSpace
         * @brief Represents the input space for the discrete wavelet transform.
         */
        class InputSpace : public Transform::InputSpace
        {
        public:
            Typedefs::vec data; /**< The input data vector. */

            /**
             * @brief Constructor for the InputSpace class.
             * @param _signal The input signal.
             */
            InputSpace(const Typedefs::vec &_signal);

            /**
             * @brief Get the input data.
             * @return The input data as a vector.
             */
            auto get_data() const -> Typedefs::vec override;
        };

        /**
         * @class OutputSpace
         * @brief Represents the output space for the discrete wavelet transform.
         */
        class OutputSpace : public Transform::OutputSpace
        {
        public:
            Typedefs::vec data; /**< The output data vector. */

            /**
             * @brief Get the plottable representation of the output data.
             * @return The plottable representation as a vector.
             */
            auto get_plottable_representation() const -> Typedefs::vec override;

            /**
             * @brief Compress the output data using a specified method and compression value.
             * @param method The compression method.
             * @param removed The compression value indicating the amount of data to remove.
             */
            auto compress(const std::string &method, const double removed) -> void override;
        };

    public:
        /**
         * @brief Constructor for the DiscreteWaveletTransform class.
         * @tparam N The size of the transform matrix.
         * @param scaling_matrix The TransformMatrix object containing the transform matrices.
         * @param user_levels The number of levels to be used in the transform (default: 0).
         * @param n_cores The number of cores to be used for parallel computation (default: -1, meaning automatic detection).
         */
        template <std::size_t N>
        DiscreteWaveletTransform(const TRANSFORM_MATRICES::TransformMatrix<Typedefs::DType, N> &scaling_matrix,
                                 uint8_t user_levels = 0, int n_cores = -1)
            : transform_matrix(scaling_matrix.forward), inverse_matrix(scaling_matrix.inverse),
              user_levels(user_levels), n_cores(n_cores) {}

        /**
         * @brief Get the input space for the discrete wavelet transform.
         * @param v The input data.
         * @return A unique pointer to the InputSpace object.
         */
        auto get_input_space(const Typedefs::vec &v) const -> std::unique_ptr<Transform::InputSpace> override;

        /**
         * @brief Get the output space for the discrete wavelet transform.
         * @return A unique pointer to the OutputSpace object.
         */
        auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> override;

        /**
         * @brief Set the number of cores to be used for parallel computation.
         * @param n_cores The number of cores.
         */
        auto set_n_cores(const int n_cores) -> void { this->n_cores = n_cores; }

        /**
         * @brief Perform the discrete wavelet transform operation on a signal vector.
         * @param signal The input signal vector.
         * @param is_inverse A flag indicating whether to perform the inverse transform.
         */
        auto operator()(Typedefs::vec &signal, const bool is_inverse) const -> void;

        /**
         * @brief Perform the discrete wavelet transform operation.
         * @param in The input space for the transformation.
         * @param out The output space for the transformation.
         * @param inverse A flag indicating whether to perform the inverse transform.
         */
        auto operator()(Transform::InputSpace &in, Transform::OutputSpace &out, const bool inverse) const -> void override;
    };
}
#endif // DISCRETE_WAVELET_TRANSFORM_HPP