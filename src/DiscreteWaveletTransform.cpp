#include <omp.h>

#include "DiscreteWaveletTransform.hpp"
#include "bitreverse.hpp"

#if USE_CUDA==1
#include "DiscreteWaveletTransformCUDA.cuh"
#endif

using namespace Typedefs;
using namespace tr;

DiscreteWaveletTransform::InputSpace::InputSpace(const vec& _signal) : data(_signal) {}

auto DiscreteWaveletTransform::InputSpace::get_data() const -> vec { return data; }

auto DiscreteWaveletTransform::OutputSpace::get_plottable_representation () const -> vec  {
    vec plottable(data);
    bitreverse::bit_reverse_copy(plottable);
    return plottable;
}

auto DiscreteWaveletTransform::OutputSpace::compress (const std::string& /*method*/, const double kept) -> void {
    auto cutoff_freq = (size_t)(kept*(double)data.size());
    bitreverse::bit_reverse_copy(data);
    for (size_t i=cutoff_freq; i<data.size(); i++){
        data[i] = 0;
    }
    bitreverse::bit_reverse_copy(data); 
}

auto DiscreteWaveletTransform::get_input_space(const vec & v) const -> std::unique_ptr<Transform::InputSpace> {
    std::unique_ptr<Transform::InputSpace> in = std::make_unique<DiscreteWaveletTransform::InputSpace>(v);
    return in;
}

auto DiscreteWaveletTransform::get_output_space() const -> std::unique_ptr<Transform::OutputSpace> {
    std::unique_ptr<Transform::OutputSpace> out = std::make_unique<DiscreteWaveletTransform::OutputSpace>();
    return out;
}

#if USE_CUDA==0
auto DiscreteWaveletTransform::operator()(std::vector<double> &signal, const bool is_inverse) const -> void{
    if (n_cores != -1) omp_set_num_threads(n_cores);

    auto& t_mat = is_inverse ? inverse_matrix : transform_matrix;
    const auto matrix_size = t_mat.size() / 2;

    std::vector<double> temp;
    auto levels = user_levels == 0 ? static_cast<int>(log2(signal.size())) : user_levels;
    auto start = is_inverse ? levels-1 : 0; 
    auto end = is_inverse ? -1 : levels;
    auto step = is_inverse ? -1 : 1;

    for (int i = start; i != end ; i += step){
        temp.clear();
        size_t sub_step = 1 << i; //pow(2, i);
        size_t sub_size = signal.size() / sub_step;

        for (size_t j = 0; j < sub_size; j++) temp.push_back(signal[j*sub_step]);
    
        if (!is_inverse) for (size_t j = 0; j < matrix_size-2; j++) temp.push_back(temp[j]);
        else             for (size_t j = 0; j < matrix_size-2; j++) temp.insert(temp.begin(), *(temp.end()-1-j));

        #pragma omp parallel for
        for(size_t j = 0; j < sub_size; j+=2){
            size_t index_signal = j*sub_step;
            signal[index_signal] = 0;
            signal[index_signal+sub_step] = 0;
            for (size_t m=0; m < matrix_size; m+=1){
                signal[index_signal]              += temp[j+m]*t_mat[m];
                signal[index_signal+sub_step]     += temp[j+m]*t_mat[m+ matrix_size];
            }
        }
    }
} 
#else 
auto DiscreteWaveletTransform::operator()(std::vector<double> &signal, const bool is_inverse) const -> void{
    dwtCU(signal, is_inverse, transform_matrix, inverse_matrix, user_levels);
}
#endif
  
auto DiscreteWaveletTransform::operator()(Transform::InputSpace& in, Transform::OutputSpace& out, const bool inverse) const -> void {
    auto& in_data = dynamic_cast<DiscreteWaveletTransform::InputSpace&>(in).data;
    auto& out_data = dynamic_cast<DiscreteWaveletTransform::OutputSpace&>(out).data;
    if (!inverse) {
        out_data = in_data;
        operator()(out_data, inverse);
    } else {
        in_data = out_data;
        operator()(in_data, inverse);
    }
}