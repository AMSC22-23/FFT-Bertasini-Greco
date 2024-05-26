#include "Image.hpp"
#include <bitreverse.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace Typedefs;

Image::Image(const cv::Mat& _img, std::shared_ptr<Transform<cv::Mat>>& _tr) : img(_img), og_size(img.size()), tr(_tr) {
    // auto is_padding_needed = n_samples & (n_samples - 1);
    // auto correct_padding = (is_padding_needed && padding) ? next_power_of_2(n_samples) : n_samples;
    
    input_space = tr->get_input_space(img);
    output_space = tr->get_output_space();
}

auto Image::inverse_transform() -> void {
    tr->operator()(*input_space, *output_space, true);
    img = input_space->get_data();
}

auto Image::preprocess_filter (const double percentile) -> double {
    cout << "Number of pixels before filtering: " << og_size.height * og_size.width;
    auto normalized_percentile = (1.0 - (1.0 - percentile) * og_size.height * og_size.width / img.total());
    cout << " vs after filtering: " << (1 - normalized_percentile) * img.total() << endl;
    return normalized_percentile;
}

auto Image::compress(const double percentile, const string method) -> void {
    auto normalized_percentile = preprocess_filter(percentile);
    output_space->compress(method, normalized_percentile);
    inverse_transform();
}

auto Image::transform() -> void {
    tr->operator()(*input_space, *output_space, false);
}

auto Image::get_image() const -> const cv::Mat {
    return img.rowRange(0, og_size.height).colRange(0, og_size.width);
}

auto Image::get_tr_coeff() const -> const cv::Mat {
    return output_space->get_plottable_representation()/*.rowRange(0, og_size.height).colRange(0, og_size.width)*/;
}

