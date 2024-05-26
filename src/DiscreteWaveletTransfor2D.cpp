#include "DiscreteWaveletTransform2D.hpp"
#include "bitreverse.hpp"

using namespace std;
using namespace cv;
using namespace Typedefs;

template <unsigned long matrix_size>
DiscreteWaveletTransform2D<matrix_size>::InputSpace::InputSpace(const cv::Mat& og_image) {
    cv::Mat image;
    og_image.convertTo(image, CV_64FC3);
    int channels = image.channels();
    int rows = image.rows;
    int cols = image.cols;
    data = Typedefs::vec3D(channels, Typedefs::vec2D(rows, Typedefs::vec(cols)));
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                data[c][i][j] = image.at<Vec3d>(i, j)[c];
}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::InputSpace::get_data() const -> cv::Mat {
    int channels = data.size();
    int rows = data[0].size();
    int cols = data[0][0].size();
    cv::Mat img_char(rows, cols, CV_64FC3);
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                img_char.at<Vec3d>(i, j)[c] = data[c][i][j];
    img_char.convertTo(img_char, CV_8UC3);
    return img_char;
}
template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::OutputSpace::bit_reverse_image(Typedefs::vec3D& image) const -> void {
    int channels = image.size();
    int rows = image[0].size();
    int cols = image[0][0].size();

    for (uint8_t l = 0; l < user_levels; l++){
        for (int c = 0; c < channels; ++c)
        {
            for (int i = 0; i < rows; ++i) partial_bit_reverse(image[c][i], cols, 1);
            for (int j=0; j<cols; j++){
                std::vector<double> temp;
                for (int i = 0; i < rows; ++i) temp.push_back(image[c][i][j]);
                partial_bit_reverse(temp, rows, 1);
                for (int i = 0; i < rows; ++i) image[c][i][j]=temp[i];
            }
        }
        rows /= 2;
        cols /= 2;
    }
}


template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::OutputSpace::get_plottable_representation() const -> cv::Mat
{
    auto tmp = data;
    bit_reverse_image(tmp);
    normalize_coefficients(tmp);
    cv::Mat dwt_image_colored;
    dwt_image_colored.create(tmp[0].size(), tmp[0][0].size(), CV_64FC3);
    for (size_t c = 0; c < tmp.size(); ++c)
        for (size_t i = 0; i < tmp[0].size(); ++i)
            for (size_t j = 0; j < tmp[0][0].size(); ++j)
                dwt_image_colored.at<cv::Vec3d>(i, j)[c] = tmp[c][i][j];
    dwt_image_colored.convertTo(dwt_image_colored, CV_8UC3);
    return dwt_image_colored;
}

template <unsigned long matrix_size>
int DiscreteWaveletTransform2D<matrix_size>::OutputSpace::countSubdivisions(int i, int j, int size, int subdivisions) const {
    int currentSize = size;

    for (int level = 0; level < subdivisions; ++level) {
        int halfSize = currentSize / 2;

        if (i < halfSize && j < halfSize) {
            // Point is in the top-left submatrix
            currentSize = halfSize;
        } else {
            // Adjust i and j for the next level of subdivision
            if (i >= halfSize) i -= halfSize;
            if (j >= halfSize) j -= halfSize;
            return subdivisions - level - 1; // Subtract 1 to make the count 0-based
        }
    }

    // If the point is not in the top-left submatrix after all subdivisions
    return 0;
} 

template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::OutputSpace::normalize_coefficients(Typedefs::vec3D& image) const -> void {
    int channels = image.size();
    int rows = image[0].size();
    int cols = image[0][0].size();

    std::vector<std::vector<double>>  max_abs_val_details(channels, std::vector<double>(user_levels+1, 0));
    std::vector<std::vector<double>>  min_abs_val_details(channels, std::vector<double>(user_levels+1, 0));

    
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
            
                uint8_t point_level = countSubdivisions(i, j, rows, user_levels+1);
                if (max_abs_val_details[c][point_level]<std::abs(image[c][i][j])){
                    max_abs_val_details[c][point_level] = std::abs(image[c][i][j]);

                }
                if (min_abs_val_details[c][point_level]>std::abs(image[c][i][j])){
                    min_abs_val_details[c][point_level] = std::abs(image[c][i][j]);
                }
            }
        }
    }

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                uint8_t point_level = countSubdivisions(i, j, rows, user_levels+1);
                image[c][i][j] = (image[c][i][j] - min_abs_val_details[c][point_level]) / (max_abs_val_details[c][point_level] - min_abs_val_details[c][point_level]) * 255;
            }
        }
    }

}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::OutputSpace::compress(const std::string& /*method*/, const double percentile) -> void {
    
    uint8_t levels_to_keep = user_levels - floor(-(log(1-percentile))/(log(4)));
    int channels = data.size();
    int rows = data[0].size();
    int cols = data[0][0].size();
     for (int c = 0; c < channels; ++c)
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols;  j++)
                if (i % static_cast<int>(pow(2, user_levels-levels_to_keep)) != 0 || j % static_cast<int>(pow(2, user_levels-levels_to_keep)) != 0 )
                        data[c][i][j] = 0;
}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::computeDWT2D(Typedefs::vec3D& image, bool is_inverse) const -> void {
    int channels = image.size();
    int rows = image[0].size();
    int cols = image[0][0].size();

    int start = is_inverse ? user_levels - 1 : 0;
    int end = is_inverse ? -1 : user_levels;
    int step = is_inverse ? -1 : 1;

    for (int l = start; l!=end; l+=step) {
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < rows; i+=pow(2,l)){
                std::vector<double> temp;
                for (int j = 0; j < cols; j+=pow(2,l))
                {
                    temp.push_back(image[c][i][j]);

                }
                dwt(temp, is_inverse);
                // DWT_1D(temp, temp.size(), transform_matrix, false, 1);
                int k = 0;
                for (int j = 0; j < cols; j+=pow(2,l))
                {
                    image[c][i][j]=temp[k];
                    k++;
                }
            }
 
        for (int c = 0; c < channels; ++c)
        {
            for (int j=0; j<cols; j+=pow(2, l)){ 
                std::vector<double> temp;
                for (int i = 0; i < rows; i+=pow(2,l))
                {
                    temp.push_back(image[c][i][j]);

                }
                dwt(temp, is_inverse);
                // DWT_1D(temp, temp.size(), transform_matrix, false, 1);
                int k = 0;
                for (int i = 0; i < rows; i+=pow(2,l))
                {
                    image[c][i][j]=temp[k];
                    k++;
                }
            }
        }   
    } 
}


template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::get_input_space(const cv::Mat& og_image) const -> std::unique_ptr<Transform::InputSpace> {
    cv::Mat image;

    auto is_padding_needed_row = og_image.rows & ((1 << user_levels) - 1);
    auto is_padding_needed_col = og_image.cols & ((1 << user_levels) - 1);

    auto correct_padding_row = (is_padding_needed_row) ? next_multiple_of_levels(og_image.rows, user_levels) : og_image.rows;
    auto correct_padding_col = (is_padding_needed_col) ? next_multiple_of_levels(og_image.cols, user_levels) : og_image.cols;

    cv::copyMakeBorder(og_image, image, 0, correct_padding_row - og_image.rows, 0, correct_padding_col - og_image.cols, cv::BORDER_CONSTANT, cv::Scalar(0));

    std::unique_ptr<Transform::InputSpace> in = std::make_unique<DiscreteWaveletTransform2D::InputSpace>(image);
    return in;
}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::get_output_space() const -> std::unique_ptr<Transform::OutputSpace> {
    std::unique_ptr<Transform::OutputSpace> out = std::make_unique<DiscreteWaveletTransform2D::OutputSpace>(user_levels);
    return out;
}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform2D<matrix_size>::operator()(Transform::InputSpace& in, Transform::OutputSpace& out, bool inverse) const -> void {
    auto& in_data = dynamic_cast<DiscreteWaveletTransform2D::InputSpace&>(in).data;
    auto& out_data = dynamic_cast<DiscreteWaveletTransform2D::OutputSpace&>(out).data;
    if (!inverse) {
        out_data = in_data;
        computeDWT2D(out_data, inverse);
    } else {
        in_data = out_data;
        computeDWT2D(in_data, inverse);
    }
}

template class DiscreteWaveletTransform2D<4>;