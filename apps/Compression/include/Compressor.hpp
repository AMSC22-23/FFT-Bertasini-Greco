#ifndef COMPRESSOR_HPP
#define COMPRESSOR_HPP

#include <typedefs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

class Compressor {
private:
    cv::Mat img;
    Typedefs::vec3D coeff;
    int levels;
    auto apply_dwt() -> void;
    auto quantize_value(double& value, const double& step) -> void;
    auto quantize () -> void;
    auto HuffmanEncoding(const std::string& filename) -> void;
public:
    auto compress(const std::string& filename, const cv::Mat& img, const int levels = 3) -> void;
};

#endif