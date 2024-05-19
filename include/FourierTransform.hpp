#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <typedefs.hpp>
#include <Transform.hpp>

class FourierTransform : public Transform<Typedefs::vec>{
    protected:
    class InputSpace : public Transform::InputSpace
    {
    public:
        Typedefs::vcpx data;
        InputSpace(const Typedefs::vec& _signal);
        auto get_data() const -> Typedefs::vec override;
    };
    class OutputSpace : public Transform::OutputSpace
    {
    public:
        Typedefs::vcpx data;
        auto get_plottable_representation () const -> Typedefs::vec override;
        auto compress (const std::string& method, const double kept) -> void override;
        auto filter_freqs(double percentile_cutoff) -> void;
        auto filter_magnitude(double percentile_cutoff) -> void;
        auto denoise(double freq_cutoff) -> void;
    };
    public:
    auto get_input_space(const Typedefs::vec & v) const -> std::unique_ptr<Transform::InputSpace> override;
    auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> override;

    auto operator()(Transform::InputSpace& in, Transform::OutputSpace& out, bool inverse) const -> void override;
    
    virtual auto operator()(Typedefs::vcpx&, const bool) const -> void = 0;
};

#endif