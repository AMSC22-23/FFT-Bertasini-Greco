#include "DiscreteWaveletTransform.hpp"
#include "bitreverse.hpp"
#include <omp.h>

#if USE_CUDA==1
#include "DiscreteWaveletTransformCUDA.cuh"
#endif

using namespace Typedefs;

DiscreteWaveletTransform::InputSpace::InputSpace(const vec& _signal) : data(_signal) {}

auto DiscreteWaveletTransform::InputSpace::get_data() const -> vec { return data; }

auto DiscreteWaveletTransform::OutputSpace::get_plottable_representation () const -> vec  {
    vec plottable(data);
    bit_reverse_copy(plottable);
    return plottable;
}

auto DiscreteWaveletTransform::OutputSpace::compress (const std::string& /*method*/, const double kept) -> void {
    size_t cutoff_freq = (size_t)(kept*(double)data.size());
    bit_reverse_copy(data);
    for (size_t i=cutoff_freq; i<data.size(); i++){
        data[i] = 0;
    }
    bit_reverse_copy(data); 
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
auto DiscreteWaveletTransform::operator()(std::vector<double> &signal, bool is_inverse) const -> void{
    if (n_cores != -1) omp_set_num_threads(n_cores);

    auto& t_mat = is_inverse ? inverse_matrix : transform_matrix;
    const unsigned long matrix_size = t_mat.size() / 2;

    std::vector<double> temp;
    int levels = user_levels == 0 ? log2(signal.size()) : user_levels;
    int start = is_inverse ? levels-1 : 0; 
    int end = is_inverse ? -1 : levels;
    int step = is_inverse ? -1 : 1;

    for (int i = start; i!=end ; i+=step){
        temp.clear();
        int sub_step = pow(2, i);
        int sub_size = signal.size()/sub_step;

        for (int j = 0; j < sub_size; j++) temp.push_back(signal[j*sub_step]);
    
        if (!is_inverse) for (unsigned long j = 0; j < matrix_size-2; j++) temp.push_back(temp[j]);
        else             for (unsigned long j = 0; j < matrix_size-2; j++) temp.insert(temp.begin(), *(temp.end()-1-j));

        #pragma omp parallel for
        for(int j = 0; j < sub_size; j+=2){
            int index_signal = j*sub_step;
            signal[index_signal] = 0;
            signal[index_signal+sub_step] = 0;
            for (unsigned long m=0; m < matrix_size; m+=1){
                signal[index_signal]              += temp[j+m]*t_mat[m];
                signal[index_signal+sub_step]     += temp[j+m]*t_mat[m+ matrix_size];
            }
        }
    }
} 
#else 
auto DiscreteWaveletTransform::operator()(std::vector<double> &signal, bool is_inverse) const -> void{
    dwtCU(signal, is_inverse, transform_matrix, inverse_matrix, user_levels);
}
#endif
  
auto DiscreteWaveletTransform::operator()(Transform::InputSpace& in, Transform::OutputSpace& out, bool inverse) const -> void {
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