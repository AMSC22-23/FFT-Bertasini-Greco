/**
 * @file DiscreteWaveletTransform2D.hpp
 * @brief Defines the DiscreteWaveletTransform2D class, which inherits from the Transform class for 2D discrete wavelet transform operations.
 */

#ifndef DWT2D_HPP
#define DWT2D_HPP

#include <opencv2/opencv.hpp>

#include "DiscreteWaveletTransform.hpp"

namespace tr
{

    /**
     * @class DiscreteWaveletTransform2D
     * @brief Represents a 2D discrete wavelet transform operation.
     * @tparam cv::Mat The matrix type used for input and output data.
     */
    class DiscreteWaveletTransform2D : public Transform<cv::Mat>
    {
    private:
        uint8_t user_levels = 0;      /**< The number of levels to be used in the transform. */
        DiscreteWaveletTransform dwt; /**< The 1D discrete wavelet transform object. */

    public:
        /**
         * @class InputSpace
         * @brief Represents the input space for the 2D discrete wavelet transform.
         */
        class InputSpace : public Transform::InputSpace
        {
        public:
            Typedefs::vec3D data; /**< The input data as a 3D vector. */

            /**
             * @brief Constructor for the InputSpace class.
             * @param og_image The original input image.
             */
            InputSpace(const cv::Mat &og_image);

            /**
             * @brief Get the input data.
             * @return The input data as a matrix.
             */
            auto get_data() const -> cv::Mat override;
        };

        /**
         * @class OutputSpace
         * @brief Represents the output space for the 2D discrete wavelet transform.
         */
        class OutputSpace : public Transform::OutputSpace
        {
        public:
            Typedefs::vec3D data; /**< The output data as a 3D vector. */
            uint8_t user_levels;  /**< The number of levels used in the transform. */

            /**
             * @brief Constructor for the OutputSpace class.
             * @param user_levels The number of levels used in the transform.
             */
            OutputSpace(const uint8_t user_levels) : user_levels(user_levels) {}

            /**
             * @brief Normalize the wavelet coefficients of the output data.
             * @param image The output data as a 3D vector.
             */
            auto normalize_coefficients(Typedefs::vec3D &image) const -> void;

            /**
             * @brief Get the plottable representation of the output data.
             * @return The plottable representation as a matrix.
             */
            auto get_plottable_representation() const -> cv::Mat override;

            /**
             * @brief Compress the output data using a specified method and compression value.
             * @param method The compression method.
             * @param kept The compression value indicating the amount of data to keep.
             */
            auto compress(const std::string &method, const double kept) -> void override;
        };

        /**
         * @brief Constructor for the DiscreteWaveletTransform2D class.
         * @tparam N The size of the transform matrix.
         * @param scaling_matrix The TransformMatrix object containing the transform matrices.
         * @param user_levels The number of levels to be used in the transform (default: 0).
         * @param n_cores The number of cores to be used for parallel computation (default: -1, meaning automatic detection).
         */
        template <std::size_t N>
        DiscreteWaveletTransform2D(const TRANSFORM_MATRICES::TransformMatrix<Typedefs::DType, N> &scaling_matrix,
                                   const uint8_t user_levels = 0, const int n_cores = -1)
            : user_levels(user_levels), dwt(scaling_matrix, 1, n_cores) {}

        /**
         * @brief Get the input space for the 2D discrete wavelet transform.
         * @param og_image The original input image.
         * @return A unique pointer to the InputSpace object.
         */
        auto get_input_space(const cv::Mat &og_image) const -> std::unique_ptr<Transform::InputSpace> override;

        /**
         * @brief Get the output space for the 2D discrete wavelet transform.
         * @return A unique pointer to the OutputSpace object.
         */
        auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> override;

        /**
         * @brief Perform the 2D discrete wavelet transform operation on a 3D vector of wavelet coefficients.
         * @param dwt_coeff The input wavelet coefficients as a 3D vector.
         * @param is_inverse A flag indicating whether to perform the inverse transform.
         */
        auto operator()(Typedefs::vec3D &dwt_coeff, const bool is_inverse) const -> void;

        /**
         * @brief Perform the 2D discrete wavelet transform operation.
         * @param in The input space for the transformation.
         * @param out The output space for the transformation.
         * @param inverse A flag indicating whether to perform the inverse transform.
         */
        auto operator()(Transform::InputSpace &in, Transform::OutputSpace &out, const bool inverse) const -> void override;
    };
}

#endif // DWT2D_HPP