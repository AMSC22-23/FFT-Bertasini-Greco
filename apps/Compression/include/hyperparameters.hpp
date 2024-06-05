#pragma once

namespace compression {
    const double compression_coeff = 1; // the higher the more compression
    const double R = 8.;
    const double c = 8.5; // exponent
    const double f = 8.;   // mantissa
    const auto tr = TRANSFORM_MATRICES::DAUBECHIES_D10;
};