#ifndef DWT2D_HPP
#define DWT2D_HPP

#include <opencv2/opencv.hpp>

#include "DiscreteWaveletTransform.hpp"

class DiscreteWaveletTransform2D : public Transform<cv::Mat> {
    private:
    uint8_t user_levels = 0;
    DiscreteWaveletTransform dwt;
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
        OutputSpace(const uint8_t user_levels) : user_levels(user_levels) {}
        auto normalize_coefficients(Typedefs::vec3D& image) const -> void;
        auto get_plottable_representation() const -> cv::Mat override;
        auto compress(const std::string& method, const double kept) -> void override;
    };
    
    template<std::size_t N> DiscreteWaveletTransform2D(const TRANSFORM_MATRICES::TransformMatrix<Typedefs::DType, N> & scaling_matrix, const uint8_t user_levels = 0, const int n_cores=-1) : user_levels(user_levels), dwt(scaling_matrix, 1, n_cores) {}
    auto get_input_space(const cv::Mat& og_image) const -> std::unique_ptr<Transform::InputSpace> override;
    auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> override ;
    auto operator()(Typedefs::vec3D& dwt_coeff, const bool is_inverse) const -> void;
    auto operator()(Transform::InputSpace& in, Transform::OutputSpace& out, const bool inverse) const -> void override;
};

#endif