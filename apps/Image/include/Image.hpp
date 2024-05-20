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
    std::shared_ptr<Transform<cv::Mat>> fft;
    std::unique_ptr<Transform<cv::Mat>::InputSpace> input_space;
    std::unique_ptr<Transform<cv::Mat>::OutputSpace> output_space;
    auto inverse_transform_signal() -> void;
    auto preprocess_filter(const double percentile) -> double;
    public:
    Image(const cv::Mat& img, std::shared_ptr<Transform<cv::Mat>>& fft);
    auto transform_signal() -> void;
    auto filter_magnitude(const double percentile) -> void;
    auto filter_freqs(const double percentile) -> void;
    auto get_image() const -> const cv::Mat;
    auto get_fft_freqs() const -> const cv::Mat;
};

#endif