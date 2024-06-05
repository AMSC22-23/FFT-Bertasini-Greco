/**
 * @file Image.hpp
 * @brief Defines the Image class for image transformation and filtering.
 */

#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <typedefs.hpp>
#include <FourierTransform.hpp>
#include <memory>
#include <opencv2/opencv.hpp>

/**
 * @brief The Image class represents an image and provides methods for transformation, compression, and retrieval.
 */
class Image {
private:
    cv::Mat img; /**< The OpenCV matrix representing the image. */
    cv::Size og_size; /**< The original size of the image. */
    std::unique_ptr<Transform<cv::Mat>> tr; /**< A unique pointer to the transformation object. */
    std::unique_ptr<Transform<cv::Mat>::InputSpace> input_space; /**< A unique pointer to the input space object. */
    std::unique_ptr<Transform<cv::Mat>::OutputSpace> output_space; /**< A unique pointer to the output space object. */

    /**
     * @brief Performs the inverse transformation on the image.
     */
    auto inverse_transform() -> void;

    /**
     * @brief Preprocesses the filter based on the given percentile.
     * @param percentile The percentile value for preprocessing.
     * @return The preprocessed filter value.
     */
    auto preprocess_filter(const double percentile) -> double;

public:
    /**
     * @brief Constructs an Image object.
     * @param img The OpenCV matrix representing the image.
     * @param tr The unique pointer to the transformation object.
     */
    Image(const cv::Mat& img, std::unique_ptr<Transform<cv::Mat>> tr);

    /**
     * @brief Performs the transformation on the image.
     */
    auto transform() -> void;

    /**
     * @brief Compresses the image using the specified method and percentile.
     * @param percentile The percentile value for compression.
     * @param method The compression method to be used.
     */
    auto compress(const double percentile, const std::string& method) -> void;

    /**
     * @brief Retrieves the image as an OpenCV matrix.
     * @return The OpenCV matrix representing the image.
     */
    auto get_image() const -> const cv::Mat;

    /**
     * @brief Retrieves the transformation coefficients.
     * @return The OpenCV matrix containing the transformation coefficients.
     */
    auto get_tr_coeff() const -> const cv::Mat;
};

#endif