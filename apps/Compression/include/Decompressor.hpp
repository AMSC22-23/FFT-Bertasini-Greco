/**
* @file Decompressor.hpp
* @brief Defines the Decompressor class for decompressing images compressed using the discrete wavelet transform and Huffman encoding.
*/

#ifndef DECOMPRESSOR_HPP
#define DECOMPRESSOR_HPP

#include <opencv2/opencv.hpp>
#include "typedefs.hpp"

/**
* @class Decompressor
* @brief A class for decompressing images compressed using the discrete wavelet transform and Huffman encoding.
*/
class Decompressor {
private:
   cv::Mat img; /**< The decompressed image. */
   cv::Size img_size; /**< The size of the decompressed image. */
   Typedefs::vec3D coeff; /**< The wavelet coefficients of the decompressed image. */
   int levels; /**< The number of levels used for the wavelet transform during compression. */

   /**
    * @brief Apply the inverse discrete wavelet transform to the wavelet coefficients.
    */
   auto apply_idwt() -> void;

   /**
    * @brief Dequantize the wavelet coefficients of the decompressed image.
    */
   auto dequantize() -> void;

   /**
    * @brief Perform Huffman decoding on the compressed data read from a file.
    * @param filename The name of the file containing the compressed data.
    */
   auto HuffmanDecoding(const std::string& filename) -> void;

public:
   /**
    * @brief Decompress an image from a compressed file.
    * @param filename The name of the file containing the compressed data.
    * @param img The output decompressed image (output parameter).
    */
   auto decompress(const std::string& filename, cv::Mat& img) -> void;
};

#endif