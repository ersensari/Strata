// JetBrains Mellum2.1: Q/K RMS norm, alternating SWA/full NeoX attention and SiLU MoE.
#pragma once
#include "engine.hpp"
namespace strata::poly {
class Mellum final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;
protected:
    void load_geometry() override;
    void begin_pass() override;
    void load_layer(int il) override;
    size_t recurrent_floats() const override { return 0; }
    void save_layer(int, int64_t, uint8_t*) override {}
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return n_head_ * head_dim_; }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer&, int, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer&, int, int n) override;
    int head_key(int, int n) const override { return room_key(n); }
    bool batch_heads() const override { return true; }
private:
    struct Block {
        Dev wq, wk, wv, wo, q_norm, k_norm;
        uint16_t *keys = nullptr, *vals = nullptr;
        bool swa = false;
    };
    void attention(Block& B, int n);
    void rotate(Block& B, float* input, float* output, const Dev& norm, int heads,
                int64_t pos, int n, const long long* at);
    std::vector<Block> blocks_;
    std::vector<uint8_t> swa_;
    std::vector<float> yarn_;
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, window_ = 0, chunk_ = 1;
    float theta_full_ = 0, theta_swa_ = 0, magnitude_ = 1, expert_scale_ = 0;
    float *freq_ = nullptr, *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr,
          *v_ = nullptr, *att_ = nullptr, *score_ = nullptr, *awork_ = nullptr;
    size_t score_n_ = 0;
};
}  // namespace strata::poly
