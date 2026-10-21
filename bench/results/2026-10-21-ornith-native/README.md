# Ornith on the original native Strata engine

2026-10-21, same Ryzen 9900X / 96 GB / RTX 5060 Ti 16 GB / CUDA 13 host as the Poly integration.
Only host GPU 1 was exposed to the test container. Both UI production containers were stopped before GPU
tests; native and Poly workers were started sequentially and cleaned up between them. No model download,
GGUF edit, router change or upstream PR was performed.

## Change and correctness checks

The original `strata` dispatches Ornith's actual `qwen35moe` architecture into `src/qwen35/q36_engine.cu`.
All 31 shards and 899 tensor layouts passed. Initial load failed because the native grouped GPU expert
format list did not accept IQ3_S/Q4_K down projections. The opt-in `STRATA_ORNITH_NATIVE` build adds these
down types and grows CPU intermediate scratch for FF 1024. Default builds retain their previous behavior.

- [Real expert parity](expert-parity.txt): trunk layers 0 and 47 (IQ3_XXS/IQ3_S), MTP layer 48 (Q4_K/Q4_K),
  expert 7, three input tokens each, real 3072/1024 geometry. Zero failed checks against independent ggml
  float reference/CPU/GPU paths. The GPU weight dequantizers matched ggml's exactly for these samples;
  activation quantization still introduces numerical error. This is not full-model logit parity.
- [Existing format regression checks](existing-pairs-parity.txt): default 2560/640 synthetic
  IQ3_XXS/IQ4_NL and Q4_K/Q5_1 passed. No full Qwen-model generation was rerun with this opt-in binary.
- Original support rejection was observed before adding the option. The reference tests retain their
  original default geometry; an additional executable compiles them at Ornith's dimensions.

## Short generation comparison

Prompt, verbatim: `bir sesli asistan yapsaydin mimariyi nasil kurardin`.
No system message, thinking off, maximum output 1024 tokens, 32768 context capacity for both engines.
Each engine received two sequential requests. This is a small exploratory sample, not a controlled
throughput/quality study. Decode speed excludes prompt processing.

Native settings: `--chunk 512 --mtp 3 --expert-vram 6.4`, FP32 KV, `STRATA_Q8_PACKED=0`.
Poly settings: the deployed IQ3 fast-attention/CPU-IQ-reuse config, Q8 KV, its normal adaptive MTP/lookup.
They therefore do **not** have equivalent KV/cache-placement/draft policies. Both use the same GGUF files.

| Request mode | Native run 1 / run 2 | Poly run 1 / run 2 |
| --- | --- | --- |
| Temperature 0, presence penalty 1.5 | 24.3 / 26.5 tok/s | 34.5 / 38.7 tok/s |
| Temperature 0, penalties off | 18.2 / 22.1 tok/s | 34.8 / 37.9 tok/s |

In the presence-penalty experiment native used 12 workers plus its caller (13 CPU threads). A short
reference-test compilation overlapped part of its first generation, so **do not treat that first value
as an uncontended measurement**. The second did not overlap a build. Poly used its usual 12-thread setting.
The greedy experiment used 11 native workers plus its caller (12 total), with no concurrent compilation.

Native MTP is already implemented, but its serve loop skips it when `Sampling::needs_logits()` is true:
nonzero temperature or penalties. Thus the first row offered no native drafts. With penalties explicitly
off, the second row **did** execute MTP: 302/1023 accepted in run 1, 271/792 in run 2. This establishes
that a new architecture name is not needed to activate MTP. It does not establish a speedup.

All requests stopped naturally except greedy Poly run 1, which reached the 1024-token output cap.
Outputs differ between engines and also between repeats. CPU/GPU placement and numerical paths can differ;
the cause of the repeated-generation divergence was not isolated in this experiment. This record does not
claim token identity or coding-quality equivalence. See raw
[presence-penalty results](presence-penalty-results.json) and [greedy results](greedy-results.json).

## Memory observations and decision

The initial native load retained optional packed Q8 decode copies: 10.18 GiB of dense allocations left
insufficient space for hybrid buffers at 32K. Disabling those lossless optional copies reduced dense
allocation to 6.43 GiB and allowed startup. The [native engine log](native-greedy-engine.txt) reports
2.09 GiB / 12 experts per layer in VRAM, 44.69 GiB pinned host expert memory and 2.53 GiB prompt staging.
The FP32 caches/buffers further consume VRAM. Native startup took around 55–59 seconds before warmup.

On this configuration the original native path works, but is slower and lacks the Poly path's Q8 KV/256K
operating behavior. Leave the normal UI on the original deployed Poly IQ3 config. Retain this experiment
behind its build option for future memory/sampling-MTP work; do not present it as a production improvement.
