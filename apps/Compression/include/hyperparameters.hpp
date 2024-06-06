#pragma once

#include "typedefs.hpp"

namespace compression {
    const Typedefs::DType compression_coeff = 1; // the higher the more compression
    const Typedefs::DType R = 8.;
    const Typedefs::DType c = 8.5; // exponent
    const Typedefs::DType f = 8.;   // mantissa
    const auto tr_mat = TRANSFORM_MATRICES::DAUBECHIES_D20;
};