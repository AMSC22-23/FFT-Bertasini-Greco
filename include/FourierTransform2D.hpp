/**
 * @file FourierTransform2D.hpp
 * @brief Defines the FourierTransform2D class for 2D Fourier transform operations.
 */

#ifndef FFT2D_HPP
#define FFT2D_HPP

#include <opencv2/opencv.hpp>

#include "DiscreteFourierTransform.hpp"
#include "IterativeFastFourierTransform.hpp"
#include "RecursiveFastFourierTransform.hpp"

namespace tr
{

    /**
     * @class FourierTransform2D
     * @brief Represents a 2D Fourier transform operation.
     * @tparam FT The type of Fourier transform to be used (must be a derived class of FourierTransform).
     */
    template <class FT>
        requires std::is_base_of_v<FourierTransform, FT>
    class FourierTransform2D : public Transform<cv::Mat>
    {
    private:
        FT ft; /**< The Fourier transform object. */

    public:
        /**
         * @class InputSpace
         * @brief Represents the input space for the 2D Fourier transform.
         */
        class InputSpace : public Transform::InputSpace
        {
        public:
            Typedefs::vcpx3D data; /**< The input data as a 3D complex vector. */

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
         * @brief Represents the output space for the 2D Fourier transform.
         */
        class OutputSpace : public Transform::OutputSpace
        {
        public:
            Typedefs::vcpx3D data; /**< The output data as a 3D complex vector. */

            /**
             * @brief Rearrange the quadrants of the output data to be centered.
             * @param data_to_arrange The output data to be rearranged.
             */
            void rearrenge_quadrants_to_center(Typedefs::vcpx3D &data_to_arrange) const;

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

            /**
             * @brief Apply a pass filter to the output data.
             * @param cutoff_perc The cutoff percentage for the filter.
             * @param is_high_pass A flag indicating whether to apply a high-pass or low-pass filter.
             */
            void pass_filter(const double cutoff_perc, const bool is_high_pass);

            /**
             * @brief Apply a magnitude filter to the output data.
             * @param cutoff_percentage The cutoff percentage for the magnitude filter.
             */
            void magnitude_filter(const double cutoff_percentage);
        };

        /**
         * @brief Compute the 2D Fourier transform on a 3D complex vector.
         * @param fft_coeff The input data as a 3D complex vector.
         * @param is_inverse A flag indicating whether to perform the inverse transform.
         */
        auto compute2DFFT(Typedefs::vcpx3D &fft_coeff, const bool is_inverse) const -> void;

        /**
         * @brief Get the input space for the 2D Fourier transform.
         * @param og_image The original input image.
         * @return A unique pointer to the InputSpace object.
         */
        auto get_input_space(const cv::Mat &og_image) const -> std::unique_ptr<Transform::InputSpace> override;

        /**
         * @brief Get the output space for the 2D Fourier transform.
         * @return A unique pointer to the OutputSpace object.
         */
        auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> override;

        /**
         * @brief Perform the 2D Fourier transform operation.
         * @param in The input space for the transformation.
         * @param out The output space for the transformation.
         * @param inverse A flag indicating whether to perform the inverse transform.
         */
        auto operator()(Transform::InputSpace &in, Transform::OutputSpace &out, const bool inverse) const -> void override;
    };
}

#endif