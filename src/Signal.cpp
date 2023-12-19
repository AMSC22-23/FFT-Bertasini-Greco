#include <Signal.hpp>
#include <bitreverse.hpp>

using namespace std;

Signal::Signal(vector<double> _freqs, vector<double> _amps, size_t n_samples, shared_ptr<FourierTransform>& fft, bool padding) : fft(fft)
{
    move(_freqs.begin(), _freqs.end(), back_inserter(this->freqs));
    move(_amps.begin(), _amps.end(), back_inserter(this->amps));

    auto is_padding_needed = n_samples & (n_samples - 1);
    auto correct_padding = (is_padding_needed && padding) ? next_power_of_2(n_samples) : n_samples;

    this->x.resize(n_samples * 2, 0);
    this->signal.resize(correct_padding, 0);
    this->generate_signal(n_samples);
    this->x.resize(correct_padding);

    this->transform_signal();
    this->compute_freqs();
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
    this->transformed_signal = this->signal;
    this->fft->operator()(this->transformed_signal, false);
}

auto Signal::compute_freqs() -> void
{
    fft_freqs.resize(this->transformed_signal.size(), 0);
    transform(
        this->transformed_signal.begin(), 
        this->transformed_signal.end(), 
        fft_freqs.begin(), 
        [](cpx c){ return abs(c); }
    );
    fft_freqs.resize(fft_freqs.size() / 2);
}

auto Signal::inverse_transform_signal() -> void {
    this->signal = this->transformed_signal;
    this->fft->operator()(this->signal, true);
    transform(this->signal.begin(), this->signal.end(), this->signal.begin(), [this](cpx c){return cpx(c.real()/(double)this->signal.size()*2, c.imag()/(double)this->signal.size()*2);});
}

auto Signal::filter_freqs(size_t freq_flat) -> void
{
    if (freq_flat > this->transformed_signal.size() / 2)
        throw invalid_argument("freq_flat must be less than half the size of the transformed signal");
    if (this->transformed_signal.size() == 0)
        this->transform_signal();
    // filter high frequencies using stl transform
    transform(
        this->transformed_signal.begin(), 
        this->transformed_signal.end(), 
        this->transformed_signal.begin(), 
        [i = 0, freq_flat](cpx c) mutable { return i++ > (int)freq_flat ? 0 : c; }
    );
    this->compute_freqs();
    this->inverse_transform_signal();
}

auto Signal::get_real_signal() const -> vector<double>
{
    vector<double> real_signal(this->signal.size(), 0);
    transform(this->signal.begin(), this->signal.end(), real_signal.begin(), [](cpx c){return c.real();});
    return real_signal;
}

auto Signal::get_signal() const -> const vcpx&
{
    return this->signal;
}

auto Signal::get_x() const -> const vector<double>&
{
    return this->x;
}

auto Signal::get_transformed_signal() const -> const vcpx&
{
    return this->transformed_signal;
}

auto Signal::get_fft_freqs() const -> const vector<double>&
{
    return this->fft_freqs;
}

