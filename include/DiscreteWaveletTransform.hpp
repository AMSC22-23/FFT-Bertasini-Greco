#ifndef DISCRETE_WAVELET_TRANSFORM_HPP
#define DISCRETE_WAVELET_TRANSFORM_HPP

#include <typedefs.hpp>
#include <Transform.hpp>
#include <TransformMatrices.hpp>
#include <span>

class DiscreteWaveletTransform : public Transform<Typedefs::vec>{
private:
    std::span <const double> transform_matrix;
    std::span <const double> inverse_matrix;
    uint8_t user_levels = 0;
    int n_cores;
protected:
    class InputSpace : public Transform::InputSpace
    {
    public:
        Typedefs::vec data;
        InputSpace(const Typedefs::vec& _signal);
        auto get_data() const -> Typedefs::vec override;
    };
    class OutputSpace : public Transform::OutputSpace
    {
    public:
        Typedefs::vec data;
        auto get_plottable_representation () const -> Typedefs::vec override;
        auto compress (const std::string& method, const double removed) -> void override;
    };
public:

    template<std::size_t N> DiscreteWaveletTransform(const TRANSFORM_MATRICES::TransformMatrix<double, N> & scaling_matrix, uint8_t user_levels = 0, int n_cores=-1) : 
    transform_matrix(scaling_matrix.forward),
    inverse_matrix(scaling_matrix.inverse),
    user_levels(user_levels), 
    n_cores(n_cores) 
    {}
    
    auto get_input_space(const Typedefs::vec & v) const -> std::unique_ptr<Transform::InputSpace> override;
    auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> override;
    
    auto set_n_cores(const int n_cores) -> void { this->n_cores = n_cores; };
    
    auto operator()(std::vector<double> &signal, bool is_inverse) const -> void;
    auto operator()(Transform::InputSpace& in, Transform::OutputSpace& out, bool inverse) const -> void override;

};
#endif