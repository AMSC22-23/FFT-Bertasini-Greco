/**
* @file Compressor.hpp
* @brief Defines the Compressor class for image compression using the discrete wavelet transform and Huffman encoding.
*/

#ifndef COMPRESSOR_HPP
#define COMPRESSOR_HPP

#include <opencv2/opencv.hpp>
#include "typedefs.hpp"

/**
* @class Compressor
* @brief A class for compressing images using the discrete wavelet transform and Huffman encoding.
*/
class Compressor {
private:
   cv::Mat img; /**< The input image. */
   Typedefs::vec3D coeff; /**< The wavelet coefficients of the image. */
   cv::Size img_size; /**< The size of the input image. */
   int levels; /**< The number of levels for the wavelet transform. */

   /**
    * @brief Apply the discrete wavelet transform to the input image.
    */
   auto apply_dwt() -> void;

   /**
    * @brief Quantize a value using a given step size.
    * @param value The value to be quantized (input/output).
    * @param step The step size for quantization.
    */
   auto quantize_value(Typedefs::DType& value, const Typedefs::DType& step) -> void;

   /**
    * @brief Quantize the wavelet coefficients of the image.
    */
   auto quantize() -> void;

   /**
    * @brief Perform Huffman encoding on the quantized wavelet coefficients and write the compressed data to a file.
    * @param filename The name of the file to write the compressed data to.
    */
   auto HuffmanEncoding(const std::string& filename) -> void;

public:
   /**
    * @brief Compress an image using the discrete wavelet transform and Huffman encoding.
    * @param filename The name of the file to write the compressed data to.
    * @param img The input image.
    * @param levels The number of levels for the wavelet transform (default: 3).
    */
   auto compress(const std::string& filename, const cv::Mat& img, const int levels = 3) -> void;
};

#endif