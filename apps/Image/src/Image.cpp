#include "Image.hpp"


using namespace cv;
using namespace Typedefs;

Image::Image(const cv::Mat& img, std::shared_ptr<Transform<cv::Mat>>& fft) : img(img), fft(fft) {
    input_space = fft->get_input_space(img);
    output_space = fft->get_output_space();
}

auto Image::filter_magnitude(const double percentile) -> void {
    output_space->compress("filter_magnitude", percentile);
    img = input_space->get_data();
}

auto Image::filter_freqs(const double percentile) -> void {
    output_space->compress("filter_freqs", percentile);
    img = input_space->get_data();
}

auto Image::transform_signal() -> void {
    fft->operator()(*input_space, *output_space, false);
}

auto Image::get_image() const -> const cv::Mat& {
    return img;
}

auto Image::get_fft_freqs() const -> const cv::Mat {
    return output_space->get_plottable_representation();
}

