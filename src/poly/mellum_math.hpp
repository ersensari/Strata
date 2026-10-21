// Mellum's full-attention YaRN frequencies. SWA layers use unscaled NeoX RoPE.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace strata::poly {
inline std::vector<float> mellum_yarn_factors(int dim, float base, float factor, int original,
                                              float beta_fast, float beta_slow) {
    if (dim <= 0 || dim % 2 || !std::isfinite(base) || base <= 1 || !std::isfinite(factor) || factor < 1 ||
        original <= 0 || !std::isfinite(beta_fast) || !std::isfinite(beta_slow) || beta_fast <= 0 || beta_slow <= 0)
        throw std::runtime_error("Mellum: invalid YaRN geometry");
    const auto correction = [&](float rotations) {
        return dim * std::log(original / (rotations * 2.f * 3.14159265358979323846f)) / (2.f * std::log(base));
    };
    const float low = std::max(0.f, std::floor(correction(beta_fast)));
    const float high = std::min(float(dim - 1), std::ceil(correction(beta_slow)));
    std::vector<float> freq(dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        const float ramp = 1.f - std::clamp((i - low) / std::max(.001f, high - low), 0.f, 1.f);
        // head_norm_rope divides the unscaled frequency by this value.
        freq[i] = 1.f / ((1.f - ramp) / factor + ramp);
    }
    return freq;
}
}  // namespace strata::poly
