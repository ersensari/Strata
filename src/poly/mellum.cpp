// Arithmetic follows third_party/llama.cpp/src/models/mellum.cpp.
#include "mellum.hpp"
#include "mellum_math.hpp"
#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/poly/native_mmvq.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::poly {
void Mellum::load_geometry() {
    load_shape();
    auto need = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(std::string("Mellum: ") + message); };
    auto number = [&](const char* key, float fallback) { const auto* v = meta_in(*gg_, key); return v ? float(v->num()) : fallback; };
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(n_head_ > 0 && n_head_ <= 256 && n_kv_ > 0 && n_head_ % n_kv_ == 0, "bad attention heads");
    need(head_dim_ > 0 && head_dim_ <= 256 && head_dim_ % 4 == 0 &&
         shape_i("attention.value_length") == head_dim_, "unsupported head dimensions");
    need(int(number("rope.dimension_count", float(head_dim_))) == head_dim_, "partial rotary embedding unsupported");
    need(opt_.max_context <= shape_i("context_length"), "context exceeds the model's trained context");
    window_ = int(number("attention.sliding_window", 0));
    need(window_ >= 0, "negative sliding window");
    swa_.assign(n_layer_, 0);
    if (window_) {
        const auto* pattern = meta_in(*gg_, "attention.sliding_window_pattern");
        if (pattern && pattern->type == MetaType::ARRAY) {
            need(int(pattern->items.size()) == n_layer_, "SWA pattern must describe every layer");
            for (int il = 0; il < n_layer_; ++il) swa_[il] = pattern->items[il].u != 0;
        } else {
            const int period = pattern ? int(pattern->u) : 4;
            need(period > 0, "invalid SWA period");
            for (int il = 0; il < n_layer_; ++il) swa_[il] = (il + 1) % period != 0;
        }
    }
    const float base = shape_f("rope.freq_base"), swa_base = number("rope.freq_base_swa", base);
    need(std::isfinite(base) && base > 1 && std::isfinite(swa_base) && swa_base > 1, "bad RoPE base");
    theta_full_ = std::pow(base, -2.f / head_dim_);
    theta_swa_ = std::pow(swa_base, -2.f / head_dim_);
    const auto* scaling = meta_in(*gg_, "rope.scaling.type");
    const std::string type = scaling ? scaling->s : "none";
    need(type == "none" || type == "yarn", "only unscaled or YaRN RoPE supported");
    if (type == "yarn") {
        const float factor = number("rope.scaling.factor", 1);
        yarn_ = mellum_yarn_factors(head_dim_, base, factor, int(number("rope.scaling.original_context_length", 0)),
                                    number("rope.scaling.yarn_beta_fast", 32), number("rope.scaling.yarn_beta_slow", 1));
        // llama.cpp's default YaRN context setup cancels the kernel's extra log scaling once.
        // The remaining multiplier is 1 + 0.1*log(factor), not its square.
        magnitude_ = (1.f + .1f * std::log(factor)) * number("rope.scaling.attn_factor", 1);
    }
    expert_scale_ = number("expert_weights_scale", 0);
    need(std::isfinite(expert_scale_) && expert_scale_ >= 0, "bad expert weight scale");
    clamp_exp_.assign(size_t(n_layer_) + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}
void Mellum::begin_pass() { blocks_.assign(layers_.size(), Block{}); }
void Mellum::load_layer(int il) {
    auto& L = layers_[il]; auto& B = blocks_[il]; B.swa = swa_[il];
    auto norm = [&](const char* name, int width) { return upload(tensor(il, name), blk(il, name), width, 1, kF32); };
    auto matrix = [&](const char* name, int in, int out) { return upload(tensor(il, name), blk(il, name), in, out, kAnyMat); };
    L.attn_norm = norm("attn_norm.weight", n_embd_); L.ffn_norm = norm("ffn_norm.weight", n_embd_);
    B.q_norm = norm("attn_q_norm.weight", head_dim_); B.k_norm = norm("attn_k_norm.weight", head_dim_);
    B.wq = matrix("attn_q.weight", n_embd_, n_head_ * head_dim_);
    B.wk = matrix("attn_k.weight", n_embd_, n_kv_ * head_dim_);
    B.wv = matrix("attn_v.weight", n_embd_, n_kv_ * head_dim_);
    B.wo = matrix("attn_output.weight", n_head_ * head_dim_, n_embd_);
    L.fixed = true;
    load_moe(il, false);
}
void Mellum::alloc_caches() {
    // Full row-addressed storage also for SWA: correct checkpoint/speculative rollback without a ring snapshot.
    // The attention mask still limits SWA to the model's window. A compact ring is a future optimization.
    const size_t row = size_t(n_kv_) * head_dim_ * sizeof(uint16_t);
    for (int il = 0; il < int(blocks_.size()); ++il) if (runs(il)) {
        blocks_[il].keys = static_cast<uint16_t*>(kv_alloc(row, 1));
        blocks_[il].vals = static_cast<uint16_t*>(kv_alloc(row, 1));
    }
    if (!yarn_.empty() && !freq_) {
        freq_ = static_cast<float*>(dmalloc(yarn_.size() * sizeof(float), "Mellum YaRN"));
        ck(cudaMemcpy(freq_, yarn_.data(), yarn_.size() * sizeof(float), cudaMemcpyHostToDevice), "Mellum YaRN");
    }
    chunk_ = int(std::clamp<size_t>((size_t(16) << 20) / (size_t(n_head_) * opt_.max_context), 1, size_t(rows_)));
}
void Mellum::assign_scratch() {
    const size_t n = size_t(rows_), q = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    qraw_ = fbuf(q); q_ = fbuf(q); kraw_ = fbuf(kv); k_ = fbuf(kv); v_ = fbuf(kv); att_ = fbuf(q);
    score_n_ = std::max(size_t(chunk_) * n_head_ * opt_.max_context,
                        attend_split_floats(chunk_, n_head_, opt_.max_context, head_dim_));
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(q + q / 2 + 1 + 3 * n * n_head_);
}
void Mellum::rotate(Block& B, float* input, float* output, const Dev& norm, int heads,
                    int64_t pos, int n, const long long* at) {
    head_norm_rope(input, head_dim_, fp(norm), output, head_dim_, heads, head_dim_,
                   B.swa ? theta_swa_ : theta_full_, pos, n, rms_eps_, st_, B.swa ? nullptr : freq_, nullptr, 1, at);
    if (!B.swa && magnitude_ != 1) axpy(output, magnitude_, output, n * heads * head_dim_, false, st_);
}
void Mellum::attention(Block& B, int n) {
    lin(B.wq, xn_, qraw_, n, xq_); lin(B.wk, xn_, kraw_, n, xq_); lin(B.wv, xn_, v_, n, xq_);
    const int qwidth = n_head_ * head_dim_, kvwidth = n_kv_ * head_dim_, window = B.swa ? window_ : 0;
    const float scale = 1.f / std::sqrt(float(head_dim_));
    if (rows_of_) {
        for (int i = 0; i < n; ++i) {
            const Row& row = rows_of_[i]; const long long* at = at_ + 2 * i;
            rotate(B, qraw_ + size_t(i) * qwidth, q_ + size_t(i) * qwidth, B.q_norm, n_head_, 0, 1, at + 1);
            rotate(B, kraw_ + size_t(i) * kvwidth, k_ + size_t(i) * kvwidth, B.k_norm, n_kv_, 0, 1, at + 1);
            auto* keys = static_cast<uint16_t*>(seq_buf(row.seq, B.keys));
            auto* vals = static_cast<uint16_t*>(seq_buf(row.seq, B.vals));
            store_f16_at(k_ + size_t(i) * kvwidth, keys, kvwidth, at, 1, 0, st_);
            store_f16_at(v_ + size_t(i) * kvwidth, vals, kvwidth, at, 1, 0, st_);
            const auto keep = pos_; pos_ = row.pos; const int sized = room(1); pos_ = keep;
            attend(q_ + size_t(i) * qwidth, keys, vals, row.pos, 1, score_, chunk_, att_ + size_t(i) * qwidth,
                   head_dim_, n_head_, n_kv_, scale, st_, window, 0, false, at, sized, score_n_);
        }
    } else {
        const long long* at = recording_ ? at_ : nullptr;
        rotate(B, qraw_, q_, B.q_norm, n_head_, pos_, n, at ? at + 1 : nullptr);
        rotate(B, kraw_, k_, B.k_norm, n_kv_, pos_, n, at ? at + 1 : nullptr);
        if (at) {
            store_f16_at(k_, B.keys, kvwidth, at, n, 0, st_); store_f16_at(v_, B.vals, kvwidth, at, n, 0, st_);
        } else {
            store_f16(k_, B.keys, kvwidth, pos_, n, st_); store_f16(v_, B.vals, kvwidth, pos_, n, st_);
        }
        if (n >= kStageRows)
            attend_batch(blas_, q_, B.keys, B.vals, pos_, n, score_, size_t(chunk_) * n_head_ * opt_.max_context,
                         awork_, att_, head_dim_, n_head_, n_kv_, scale, st_, window);
        else
            attend(q_, B.keys, B.vals, pos_, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_,
                   window, 0, false, at, at ? room(n) : 0, score_n_);
    }
    lin(B.wo, att_, ao_, n);
}
void Mellum::layer_head(Layer& L, int il, int n) {
    poly_kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, sum_threads(n_embd_), st_);
    dbg("attn_in", il, xn_, n_embd_); attention(blocks_[il], n); dbg("attn", il, ao_, n_embd_);
    poly_kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, sum_threads(n_embd_), st_);
    dbg("ffn_in", il, xn_, n_embd_); if (!batched(L, n)) moe_router(L, n);
}
void Mellum::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }
void Mellum::output_rows(int first, int rows, float* h) { rms_norm(r_ + size_t(first) * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_); }
void Mellum::moe_shared(const Layer&, int, int n) { ck(cudaMemsetAsync(fo_, 0, size_t(n) * n_embd_ * sizeof(float), cs(st_)), "Mellum experts"); }
void Mellum::route(int, const float* logits, int32_t, int* ids, float* w, float* tmp) const {
    const float max = *std::max_element(logits, logits + n_expert_);
    for (int i = 0; i < n_expert_; ++i) tmp[i] = std::exp(logits[i] - max);
    float sum = 0;
    for (int k = 0; k < n_used_; ++k) {
        const int best = int(std::max_element(tmp, tmp + n_expert_) - tmp);
        ids[k] = best; w[k] = tmp[best]; sum += w[k]; tmp[best] = -1;
    }
    for (int k = 0; k < n_used_; ++k) w[k] = w[k] / std::max(sum, 6.103515625e-5f) * (expert_scale_ ? expert_scale_ : 1.f);
}
bool Mellum::route_card(int, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 0, 1, nullptr, nullptr, true, expert_scale_, ids, w, st_); return true;
}
size_t Mellum::card_row_bytes() const { return size_t(std::count_if(blocks_.begin(), blocks_.end(), [](const Block& b) { return b.keys != nullptr; })) * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }
void Mellum::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context) throw std::logic_error("Mellum: invalid saved rows");
    ck(cudaStreamSynchronize(cs(st_)), "Mellum save rows");
    const size_t row = size_t(n_kv_) * head_dim_ * sizeof(uint16_t), bytes = size_t(n) * row;
    for (const auto& B : blocks_) if (B.keys && bytes) for (const auto* cache : {B.keys, B.vals}) {
        ck(cudaMemcpy(dst, reinterpret_cast<const uint8_t*>(cache) + size_t(first) * row, bytes, cudaMemcpyDeviceToHost), "Mellum save rows"); dst += bytes;
    }
}
void Mellum::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || count < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("Mellum: invalid restored rows");
    cover(first + n); ck(cudaStreamSynchronize(cs(st_)), "Mellum load rows");
    const size_t row = size_t(n_kv_) * head_dim_ * sizeof(uint16_t), bytes = size_t(n) * row;
    for (const auto& B : blocks_) if (B.keys && bytes) for (auto* cache : {B.keys, B.vals}) {
        ck(cudaMemcpy(reinterpret_cast<uint8_t*>(cache) + size_t(first) * row, src + size_t(first - from) * row, bytes, cudaMemcpyHostToDevice), "Mellum load rows"); src += size_t(count) * row;
    }
}
}  // namespace strata::poly
