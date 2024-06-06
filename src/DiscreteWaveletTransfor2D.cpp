#include "DiscreteWaveletTransform2D.hpp"
#include "bitreverse.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;
using namespace Typedefs;
using namespace tr;

DiscreteWaveletTransform2D::InputSpace::InputSpace(const cv::Mat& og_image) {
    cv::Mat image;
    og_image.convertTo(image, CV_64FC3);
    auto channels = image.channels();
    auto rows = image.rows;
    auto cols = image.cols;
    data = Typedefs::vec3D(channels, Typedefs::vec2D(rows, Typedefs::vec(cols)));
    for (auto c = 0; c < channels; ++c)
        for (auto i = 0; i < rows; ++i)
            for (auto j = 0; j < cols; ++j)
                data[c][i][j] = image.at<Vec3d>(i, j)[c];
}

auto DiscreteWaveletTransform2D::InputSpace::get_data() const -> cv::Mat {
    auto channels = data.size();
    auto rows = data[0].size();
    auto cols = data[0][0].size();
    cv::Mat img_char(rows, cols, CV_64FC3);
    for (auto c = 0ull; c < channels; ++c)
        for (auto i = 0ull; i < rows; ++i)
            for (auto j = 0ull; j < cols; ++j)
                img_char.at<Vec3d>(i, j)[c] = data[c][i][j];
    img_char.convertTo(img_char, CV_8UC3);
    return img_char;
}

auto DiscreteWaveletTransform2D::OutputSpace::get_plottable_representation() const -> cv::Mat
{
    auto tmp = data;
    bitreverse::bit_reverse_image(tmp, user_levels);
    normalize_coefficients(tmp);
    cv::Mat dwt_image_colored;
    dwt_image_colored.create(tmp[0].size(), tmp[0][0].size(), CV_64FC3);
    for (auto c = 0ull; c < tmp.size(); ++c)
        for (auto i = 0ull; i < tmp[0].size(); ++i)
            for (auto j = 0ull; j < tmp[0][0].size(); ++j)
                dwt_image_colored.at<cv::Vec3d>(i, j)[c] = tmp[c][i][j];
    dwt_image_colored.convertTo(dwt_image_colored, CV_8UC3);
    return dwt_image_colored;
}

auto DiscreteWaveletTransform2D::OutputSpace::normalize_coefficients(Typedefs::vec3D& image) const -> void {
    auto channels = image.size();
    auto rows = image[0].size();
    auto cols = image[0][0].size();

    vec2D max_abs_val_details(channels, vec(user_levels+1, 0));
    vec2D  min_abs_val_details(channels, vec(user_levels+1, 0));

    for (auto c = 0ull; c < channels; ++c) {
        for (auto i = 0ull; i < rows; ++i) {
            for (auto j = 0ull; j < cols; ++j) {
            
                uint8_t point_level = utils::countSubdivisions(i, j, rows, cols, user_levels+1);
                if (point_level == 0) continue;
                if (max_abs_val_details[c][point_level]<std::abs(image[c][i][j])){
                    max_abs_val_details[c][point_level] = std::abs(image[c][i][j]);

                }
                if (min_abs_val_details[c][point_level]>std::abs(image[c][i][j])){
                    min_abs_val_details[c][point_level] = std::abs(image[c][i][j]);
                }
            }
        }
    }

    for (auto c = 0ull; c < channels; ++c) {
        for (auto i = 0ull; i < rows; ++i) {
            for (auto j = 0ull; j < cols; ++j) {
                uint8_t point_level = utils::countSubdivisions(i, j, rows, cols, user_levels+1);
                if (point_level == 0) continue;
                image[c][i][j] = (image[c][i][j] - min_abs_val_details[c][point_level]) / (max_abs_val_details[c][point_level] - min_abs_val_details[c][point_level]) * 255;
            }
        }
    }

}

auto DiscreteWaveletTransform2D::OutputSpace::compress(const std::string& /*method*/, const double percentile) -> void {
    
    uint8_t levels_to_keep = user_levels - floor(-(log(1-percentile))/(log(4)));
    auto channels = data.size();
    auto rows = data[0].size();
    auto cols = data[0][0].size();
     for (auto c = 0ull; c < channels; ++c)
        for (auto i = 0ull; i < rows; i++)
            for (auto j = 0ull; j < cols;  j++)
                if (i % static_cast<int>(pow(2, user_levels-levels_to_keep)) != 0 || j % static_cast<int>(pow(2, user_levels-levels_to_keep)) != 0 )
                        data[c][i][j] = 0;
}

auto DiscreteWaveletTransform2D::operator()(Typedefs::vec3D& image, bool is_inverse) const -> void {
    auto channels = image.size();
    auto rows = image[0].size();
    auto cols = image[0][0].size();

    auto start = is_inverse ? user_levels - 1 : 0;
    auto end = is_inverse ? -1 : user_levels;
    auto step = is_inverse ? -1 : 1;

    for (auto l = start; l!=end; l+=step) {
        for (auto c = 0ull; c < channels; ++c)
            for (auto i = 0ull; i < rows; i += (1ull << l) ) 
            {
                vec temp;
                for (auto j = 0ull; j < cols; j += (1ull << l) )
                {
                    temp.push_back(image[c][i][j]);

                }
                dwt(temp, is_inverse);
                // DWT_1D(temp, temp.size(), transform_matrix, false, 1);
                auto k = 0;
                for (auto j = 0ull; j < cols; j += (1ull << l) )
                {
                    image[c][i][j]=temp[k];
                    k++;
                }
            }
 
        for (auto c = 0ull; c < channels; ++c)
        {
            for (auto j=0ull; j<cols; j+=pow(2, l)){ 
                vec temp;
                for (auto i = 0ull; i < rows; i += (1ull << l))
                {
                    temp.push_back(image[c][i][j]);

                }
                dwt(temp, is_inverse);
                // DWT_1D(temp, temp.size(), transform_matrix, false, 1);
                auto k = 0;
                for (auto i = 0ull; i < rows; i += (1ull << l))
                {
                    image[c][i][j]=temp[k];
                    k++;
                }
            }
        }   
    } 
}


auto DiscreteWaveletTransform2D::get_input_space(const cv::Mat& og_image) const -> std::unique_ptr<Transform::InputSpace> {
    cv::Mat image;

    auto is_padding_needed_row = og_image.rows & ((1 << user_levels) - 1);
    auto is_padding_needed_col = og_image.cols & ((1 << user_levels) - 1);

    auto correct_padding_row = (is_padding_needed_row) ? bitreverse::next_multiple_of_levels(og_image.rows, user_levels) : og_image.rows;
    auto correct_padding_col = (is_padding_needed_col) ? bitreverse::next_multiple_of_levels(og_image.cols, user_levels) : og_image.cols;

    cv::copyMakeBorder(og_image, image, 0, correct_padding_row - og_image.rows, 0, correct_padding_col - og_image.cols, cv::BORDER_CONSTANT, cv::Scalar(0));

    std::unique_ptr<Transform::InputSpace> in = std::make_unique<DiscreteWaveletTransform2D::InputSpace>(image);
    return in;
}

auto DiscreteWaveletTransform2D::get_output_space() const -> std::unique_ptr<Transform::OutputSpace> {
    std::unique_ptr<Transform::OutputSpace> out = std::make_unique<DiscreteWaveletTransform2D::OutputSpace>(user_levels);
    return out;
}

auto DiscreteWaveletTransform2D::operator()(Transform::InputSpace& in, Transform::OutputSpace& out, const bool inverse) const -> void {
    auto& in_data  = dynamic_cast<DiscreteWaveletTransform2D::InputSpace&>(in).data;
    auto& out_data = dynamic_cast<DiscreteWaveletTransform2D::OutputSpace&>(out).data;
    if (!inverse) {
        out_data = in_data;
        operator()(out_data, inverse);
    } else {
        in_data = out_data;
        operator()(in_data, inverse);
    }
}