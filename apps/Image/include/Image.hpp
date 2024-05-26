#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <typedefs.hpp>
#include <FourierTransform.hpp>
#include <memory>
#include <opencv2/opencv.hpp>

class Image {
    private:
    cv::Mat img;
    cv::Size og_size;
    std::shared_ptr<Transform<cv::Mat>> tr;
    std::unique_ptr<Transform<cv::Mat>::InputSpace> input_space;
    std::unique_ptr<Transform<cv::Mat>::OutputSpace> output_space;
    auto inverse_transform() -> void;
    auto preprocess_filter(const double percentile) -> double;
    public:
    Image(const cv::Mat& img, std::shared_ptr<Transform<cv::Mat>>& fft);
    auto transform() -> void;
    auto compress(const double percentile, const std::string method) -> void;
    auto get_image() const -> const cv::Mat;
    auto get_tr_coeff() const -> const cv::Mat;
};

#endif