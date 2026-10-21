// CPU-only YaRN check against ggml's independent RoPE operator. No model or GPU is loaded.
#include "mellum_math.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

int main() {
    constexpr int dim = 128, heads = 2, tokens = 5, original = 8192;
    const int32_t positions[tokens] = {0, 1023, 8191, 10000, 131071};
    float worst = 0;
    for (float factor : {1.f, 16.f}) for (bool swa : {false, true}) {
        auto freq = strata::poly::mellum_yarn_factors(dim, 500000.f, factor, original, 32, 1);
        ggml_init_params params{size_t(4) << 20, nullptr, false};
        auto* ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init");
        auto* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, tokens);
        auto* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
        std::copy(positions, positions + tokens, static_cast<int32_t*>(pos->data));
        auto* data = static_cast<float*>(x->data);
        for (int i = 0; i < dim * heads * tokens; ++i) data[i] = std::sin(float(i) * .37f);
        auto* ref = ggml_rope_ext(ctx, x, pos, nullptr, dim, GGML_ROPE_TYPE_NEOX, original, 500000.f,
                                  swa ? 1.f : 1.f / factor, swa ? 0.f : 1.f, 1.f, 32.f, 1.f);
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ref);
        if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml graph");
        auto* expected = static_cast<float*>(ref->data);
        const float theta_scale = std::pow(500000.f, -2.f / dim);
        const float magnitude = swa ? 1.f : 1.f + .1f * std::log(factor);
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) for (int j = 0; j < dim / 2; ++j) {
            const int at = (t * heads + h) * dim;
            const float angle = positions[t] * std::pow(theta_scale, float(j)) / (swa ? 1.f : freq[j]);
            const float c = std::cos(angle) * magnitude, s = std::sin(angle) * magnitude;
            const float a = data[at + j] * c - data[at + j + dim / 2] * s;
            const float b = data[at + j] * s + data[at + j + dim / 2] * c;
            worst = std::max({worst, std::abs(a - expected[at + j]), std::abs(b - expected[at + j + dim / 2])});
        }
        ggml_free(ctx);
    }
    for (const float invalid : {0.f, -1.f, INFINITY}) {
        bool rejected = false;
        try { strata::poly::mellum_yarn_factors(dim, 500000.f, invalid, original, 32, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid YaRN factor accepted");
    }
    std::printf("Mellum RoPE: worst absolute error %.8g over SWA/full, factors 1/16, positions through 131071\n", worst);
    // Existing CUDA head_norm_rope computes powers of theta_scale; ggml computes base powers.
    // The high-position tolerance includes that existing floating-point difference.
    if (worst > .01f) throw std::runtime_error("Mellum RoPE differs from ggml");
}
