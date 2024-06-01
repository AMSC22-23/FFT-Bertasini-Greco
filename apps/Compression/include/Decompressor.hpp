#ifndef DECOMPRESSOR_HPP
#define DECOMPRESSOR_HPP

#include "typedefs.hpp"
#include <opencv2/opencv.hpp>

class Decompressor {
private:
    cv::Mat img;
    Typedefs::vec3D coeff;
    int levels;
    auto apply_idwt() -> void;
    auto dequantize () -> void;
    auto HuffmanDecoding(const std::string& filename) -> void;
public:
    auto decompress(const std::string& filename, cv::Mat& img) -> void;
};

#endif