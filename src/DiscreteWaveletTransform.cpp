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

    std::vector<double> temp;
    int levels = user_levels == 0 ? log2(signal.size()) : user_levels;
    int start = is_inverse ? levels-1 : 0; 
    int end = is_inverse ? -1 : levels;
    int step = is_inverse ? -1 : 1;

    for (int i = start; i!=end ; i+=step){
        temp.clear();
        int sub_step = pow(2, i);
        int sub_size = signal.size()/sub_step;

        for (int j = 0; j < sub_size; j++){
            temp.push_back(signal[j*sub_step]);
        }

        #pragma omp parallel for
        for(int j = 0; j < sub_size; j+=2){
            for (unsigned long m=0; m < matrix_size/2; m+=1){
                signal[(j+m)*sub_step] = temp[j]*transform_matrix[m*2] + temp[j+1]*transform_matrix[2*m+1];
                
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

template class DiscreteWaveletTransform<4>;