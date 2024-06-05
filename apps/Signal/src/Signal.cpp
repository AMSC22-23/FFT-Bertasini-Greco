#include <algorithm>

#include "Signal.hpp"
#include "bitreverse.hpp"
#include "utils.hpp"

using namespace std;
using namespace Typedefs;
using namespace tr;

Signal::Signal(vec _freqs, vec _amps, size_t n_samples, unique_ptr<Transform<vec>> _tr, bool padding) : tr(std::move(_tr))
{
    move(_freqs.begin(), _freqs.end(), back_inserter(this->freqs));
    move(_amps.begin(), _amps.end(), back_inserter(this->amps));

    auto is_padding_needed = n_samples & (n_samples - 1);
    auto correct_padding = (is_padding_needed && padding) ? utils::next_power_of_2(n_samples) : n_samples;

    this->x.resize(n_samples * 2, 0);
    this->signal.resize(correct_padding, 0);
    this->generate_signal(n_samples);
    this->x.resize(correct_padding);

    this->input_space  = this->tr->get_input_space(this->signal);
    this->output_space = this->tr->get_output_space();

    this->transform_signal();
}

auto Signal::generate_signal(size_t n_samples) -> void
{   
    generate(x.begin(), x.end(), [i = 0, this]() mutable {return i++ * M_PI * 4 / (double)this->x.size();});
    for (size_t i = 0; i < freqs.size(); i++)
        for (size_t n = 0; n < n_samples; n++)
            signal[n] += amps[i] * sin(freqs[i] * x[n]);
}

auto Signal::transform_signal() -> void
{
    this->tr->operator()(*this->input_space, *this->output_space, false);
}

auto Signal::denoise(const double percentile) -> void {
    this->output_space->compress("denoise", percentile);
    this->tr->operator()(*this->input_space, *this->output_space, true);
    this->signal = this->input_space->get_data();
}

auto Signal::get_signal() const -> const vec&
{
    return this->signal;
}

auto Signal::get_x() const -> const vec&
{
    return this->x;
}

auto Signal::get_freqs() const -> const Typedefs::vec
{
    return this->output_space->get_plottable_representation();
}

