#include <FourierTransform.hpp>

using namespace Typedefs;

FourierTransform::InputSpace::InputSpace(const vec& _signal) : data(_signal.size()) { 
    for (size_t i = 0; i < _signal.size(); i++) data[i] = cpx(_signal[i], 0); 
};

auto FourierTransform::InputSpace::get_data() const -> vec {
    vec signal(data.size());
    transform(data.begin(), data.end(), signal.begin(), [](cpx c){return c.real();});
    return signal;
}

auto FourierTransform::OutputSpace::get_plottable_representation () const -> vec  {
    auto& data = this->data;
    vec plottable(data.size());
    transform(data.begin(), data.end(), plottable.begin(), [](cpx c){return abs(c);});
    plottable.resize(plottable.size() / 2);
    return plottable;
}

auto FourierTransform::OutputSpace::compress (const std::string& method, const double kept) -> void {
    if (method == "filter_freqs") {
        filter_freqs(kept);
    } else if (method == "filter_magnitude") {
        filter_magnitude(kept);
    } else if (method == "denoise") {
        denoise(kept);
    } else {
        throw std::invalid_argument("Invalid compression method");
    }
}

auto FourierTransform::OutputSpace::filter_freqs(double percentile_cutoff) -> void {
    auto& data = this->data;
    size_t cutoff = percentile_cutoff * data.size();
    for (size_t i = cutoff; i < data.size(); i++) 
        data[i] = 0;            
}

auto FourierTransform::OutputSpace::filter_magnitude(double percentile_cutoff) -> void {
    auto& data = this->data;
    std::sort(data.begin(), data.end(), [](cpx a, cpx b){return abs(a) > abs(b);});
    size_t cutoff = percentile_cutoff * data.size();
    for (size_t i = cutoff; i < data.size(); i++) 
        data[i] = 0;
}

auto FourierTransform::OutputSpace::denoise(double freq_cutoff) -> void {
    auto& data = this->data;
    for (size_t i = 0; i < data.size(); i++) 
        if (i > freq_cutoff) data[i] = 0;
}

auto FourierTransform::get_input_space(const vec & v) const -> std::unique_ptr<Transform::InputSpace> {
    std::unique_ptr<Transform::InputSpace> in = std::make_unique<FourierTransform::InputSpace>(v);
    return in;
}
auto FourierTransform::get_output_space() const -> std::unique_ptr<Transform::OutputSpace> {
    std::unique_ptr<Transform::OutputSpace> out = std::make_unique<FourierTransform::OutputSpace>();
    return out;
}

auto FourierTransform::operator()(Transform::InputSpace& in, Transform::OutputSpace& out, bool inverse) const -> void {
    auto& in_data = dynamic_cast<FourierTransform::InputSpace&>(in).data;
    auto& out_data = dynamic_cast<FourierTransform::OutputSpace&>(out).data;
    if (!inverse) {
        out_data = in_data;
        operator()(out_data, inverse);
    } else {
        in_data = out_data;
        operator()(in_data, inverse);
        transform(in_data.begin(), in_data.end(), in_data.begin(), [in_data](cpx c){return cpx(c.real()/(double)in_data.size()*2, c.imag()/(double)in_data.size()*2);});
    }
}
