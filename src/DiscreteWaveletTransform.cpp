#include "DiscreteWaveletTransform.hpp"
#include "bitreverse.hpp"
#include <omp.h>

using namespace Typedefs;

template <unsigned long matrix_size>
DiscreteWaveletTransform<matrix_size>::InputSpace::InputSpace(const vec& _signal) : data(_signal) {}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform<matrix_size>::InputSpace::get_data() const -> vec { return data; }

template <unsigned long matrix_size>
auto DiscreteWaveletTransform<matrix_size>::OutputSpace::get_plottable_representation () const -> vec  {
    vec plottable(data);
    bit_reverse_copy(plottable);
    return plottable;
}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform<matrix_size>::OutputSpace::compress (const std::string& /*method*/, const double kept) -> void {
    size_t cutoff_freq = (size_t)(kept*(double)data.size());
    bit_reverse_copy(data);
    for (size_t i=cutoff_freq; i<data.size(); i++){
        data[i] = 0;
    }
    bit_reverse_copy(data); 
}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform<matrix_size>::get_input_space(const vec & v) const -> std::unique_ptr<Transform::InputSpace> {
    std::unique_ptr<Transform::InputSpace> in = std::make_unique<DiscreteWaveletTransform::InputSpace>(v);
    return in;
}

template <unsigned long matrix_size>
auto DiscreteWaveletTransform<matrix_size>::get_output_space() const -> std::unique_ptr<Transform::OutputSpace> {
    std::unique_ptr<Transform::OutputSpace> out = std::make_unique<DiscreteWaveletTransform::OutputSpace>();
    return out;
}
template <unsigned long matrix_size>
auto DiscreteWaveletTransform<matrix_size>::operator()(std::vector<double> &signal, bool is_inverse) const -> void{
    if (n_cores != -1) omp_set_num_threads(n_cores);

    auto& t_mat = is_inverse ? inverse_matrix : transform_matrix;

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
  
template <unsigned long matrix_size>
auto DiscreteWaveletTransform<matrix_size>::operator()(Transform::InputSpace& in, Transform::OutputSpace& out, bool inverse) const -> void {
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

template class DiscreteWaveletTransform<2>;
template class DiscreteWaveletTransform<4>;
template class DiscreteWaveletTransform<6>;
template class DiscreteWaveletTransform<8>;
template class DiscreteWaveletTransform<10>;
template class DiscreteWaveletTransform<16>;
template class DiscreteWaveletTransform<20>;
template class DiscreteWaveletTransform<30>;
template class DiscreteWaveletTransform<40>;
