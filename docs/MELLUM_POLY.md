# Mellum in the optional Poly engine

This fork now has an experimental `mellum` backend. It is not installed into the
running host service. Deployment and GPU inference require the owner's approval.

The local model is `/mnt/llm-models/mellum-12b/Mellum2.1-12B-A2.5B-Thinking-IQ4_XS.gguf`.
Its header declares 28 layers, 64 experts (8 used per token), width 2304, 32 query
heads and 4 KV heads of 128 values. Twenty-one layers use a 1024-token sliding
attention window; seven use full attention. The trained context is 131072 tokens.

## Implementation

- `src/poly/mellum.cpp` implements Q/K RMS normalization, NeoX rotary embedding,
  full-layer YaRN (16x from 8192), unscaled sliding-layer RoPE, SiLU experts and
  normalized top-k softmax routing, with no shared expert.
- Attention cache format is FP16. Q8 cache is not enabled for Mellum.
- `mellum2` tokenization uses the two regex passes in the bundled llama.cpp,
  rather than falling back to Qwen's regex. The GGUF supplies the vocabulary,
  merges, special tokens and chat template.
- The server selects Mellum's JSON tool-call format only for `mellum2`, preserving
  the existing Qwen/native behavior. Split streamed calls, literal closing tags in
  JSON strings, parallel calls, invalid JSON and undeclared tools have CPU tests.
- Mellum opens `<think>` itself. The parser handles that opener and the template's
  `enable_thinking` on/off switch. Reasoning levels are not a measured quality scale.

## Prepared, not deployed

The separate CUDA build is `build-mellum-staging/strata-poly` (sm_120, CUDA 13.0.48).
The active `build-poly-host/strata-poly` and native `engine/strata` were not replaced.
The service was not restarted. The build containers had no GPU access.

A private staged config and tokenizer are under `build-mellum-staging/pending/`.
They are outside the live controller's config discovery path and are git ignored.
The config selects physical GPU 1, 128K context, FP16 KV, window 1, lookup off,
no draft, no expert skipping, and frozen expert placement for initial validation.
There is no MTP block in this model. No weights were downloaded or moved.

After approval, first run a short, separate comparison against llama.cpp on the
same token IDs. Check logits/top tokens, prompt batching, the sliding-window
boundary, sequence save/restore and generated text before adopting it in the UI.
Do not claim an inference speed or end-to-end correctness from compilation alone.

## Validation and limits

CPU-only RoPE checks compare against ggml's independent RoPE operator at positions
0, 1023, 8191, 10000 and 131071, for scaling 1/16 and sliding/full layers. The
worst absolute difference was 0.0048139319 (tolerance 0.01), including the existing
`head_norm_rope` kernel's powers-of-theta-scale versus ggml's base-power rounding.
This tests the host representation of the math, not GPU kernel parity.

The first implementation keeps row-addressed KV storage even in sliding layers.
The mask is sliding, but storage is not yet a compact ring: at 128K the complete
FP16 KV allocation can reach 7 GiB, rather than about 1.8 GiB with compact rings.
Allocation grows with actual context, sharing VRAM with the expert cache. A compact
ring must preserve speculative rollback and conversation snapshots before adoption.

GPU model loading, output quality, throughput, long-context inference and comparison
against llama.cpp have not been run. HIP is not built or validated here. The local
real vocabulary was checked for code/Unicode round trips and template rendering,
including tools and thinking on/off; these are not model-generated answers.

The targeted Python suite passed 104 tests and 289 subtests. Both CPU CTest
checks passed. The broader server suite passed 350 tests and 509 subtests, with
one failing HIP ordinal test (`GpuChoice.test_hip_ordinal`). That same failure
was reproduced on an unmodified archive of the previous HEAD; it observes this
host's integrated GPU ordering. The suite also emitted a psutil mock/thread
warning in the telemetry test. Neither is counted as a passing check.
Machine-readable evidence is in
`bench/results/2026-10-22-mellum-preparation.json`.
