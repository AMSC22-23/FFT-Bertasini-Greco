#ifndef DWT2D_HPP
#define DWT2D_HPP

#include <opencv2/opencv.hpp>

#include <DiscreteWaveletTransform.hpp>

template <unsigned long matrix_size>
class DiscreteWaveletTransform2D : public Transform<cv::Mat> {
    private:
    uint8_t user_levels = 0;
    DiscreteWaveletTransform<matrix_size> dwt;
    public:
    class InputSpace : public Transform::InputSpace {
        public:
        Typedefs::vec3D data;
        InputSpace(const cv::Mat& og_image);
        auto get_data() const -> cv::Mat override;
    };
    class OutputSpace : public Transform::OutputSpace {
        public:
        Typedefs::vec3D data;
        uint8_t user_levels;
        OutputSpace(uint8_t user_levels) : user_levels(user_levels) {}
        auto normalize_coefficients(Typedefs::vec3D& image) const -> void;
        auto get_plottable_representation() const -> cv::Mat override;
        auto compress(const std::string& method, const double kept) -> void override;
    };
    
    DiscreteWaveletTransform2D(const std::array <double, matrix_size> &transform_matrix, uint8_t user_levels = 0, int n_cores=-1) : user_levels(user_levels), dwt(transform_matrix, 1, n_cores) {}
    auto computeDWT2D(Typedefs::vec3D& dwt_coeff, bool is_inverse) const -> void;
    auto get_input_space(const cv::Mat& og_image) const -> std::unique_ptr<Transform::InputSpace> override;
    auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> override ;
    auto operator()(Transform::InputSpace& in, Transform::OutputSpace& out, bool inverse) const -> void override;
};

#endif