# Poly mixed gate/up expert support

2026-10-22 (Australia/Melbourne). The owner requested support for the downloaded
Bartowski Ornith Q4_K_M, previously rejected because gate/up types differ.
This stays on `codex/poly-engines`; no pull request was opened.

## Implementation

- A layer keeps separate gate/up types, byte sizes and source strides. Fused and
  same-type tensors retain their existing combined products and byte layout.
- GPU cache sizing includes all three block types. Initial filling, growing,
  expert replacement and host memory spans address the up tensor independently.
- Cached mixed experts use separate gate and up GPU matrix-vector products,
  with the existing grouped-slots kernels where supported.
- CPU up products use their own row stride and dot-product traits. Activations
  are shared when formats match; otherwise up inputs are separately quantized
  (Q6_K gate / Q8_0 up requires Q8_K and Q8_0 inputs).
- The existing prompt MMQ and GPU-only routing paths assume one gate/up format.
  Mixed layers explicitly use the small-group GPU/CPU path, including prefill.
  Same-type layers still use their existing prompt tiles. This is a correctness
  implementation; long mixed-type prompt prefill remains an optimization target.
- The GGUF files and quantization precision were not modified. Model selection
  opts into the new format; the original native Qwen engine is unchanged.

## Verification

Built `strata-poly`, `poly_expert_parity`, and `poly_cpu_iq_test` with the existing
CUDA 13 / sm_120 build and pinned ggml. HIP was not compiled or exercised.

The expert parity tool now handles split files and gate/up/down type triples.
For each triple it checks expert 0, 128 and 255 against a double-precision
reference over ggml-dequantized weights, and grouped GPU products against single
products bit for bit. Existing numerical error tolerance .05 was not loosened.
These errors include activation quantization; they are not model quality scores.

- Q4: 7 triples x 3 experts, CPU/GPU reference checks passed. Grouped GPU results
  matched the individual products bit for bit. Includes the Q4_0 draft block and
  the mixed Q6_K/Q8_0/Q8_0 final trunk block.
- Q4 CPU: 224 cases, 1–8 rows, SiLU/GELU, finite/infinite clamp, shared and varying
  experts, comparing grouped token processing with independent rows: exact match.
- Atomic IQ3: both triples x 3 experts passed CPU/GPU parity and grouped GPU checks.
  CPU 64 cases passed with IQ reuse off (exact) and on (max relative L2 .00030556808).
- New image rebuilt; compose validation and `git diff --check` passed.

The small test logs are included here. Reproduce after building:

```sh
poly_expert_parity FIRST_SHARD.gguf --experts-only
poly_cpu_iq_test FIRST_SHARD.gguf
STRATA_POLY_IQ_MT=1 poly_cpu_iq_test ATOMIC_IQ3_FIRST_SHARD.gguf
```

## Original-app measurements

Host GPU 1: RTX 5060 Ti 16 GB; 12 engine worker threads. One model engine at a time.
The old PolyStrata container and development container are stopped. Tests used
the original Strata app at http://127.0.0.1:10110 in Manual mode, thinking off.
The same question was regenerated in separate chats:

> bir sesli asistan yapsaydin mimariyi nasil kurardin

The app's system prompt makes this 92 prompt tokens. App sampling overrides:
temperature .6, top_p .95, top_k 20, repeat_penalty 1.05, seed -1, from its
response defaults and saved topK setting. The server's remaining defaults:
min_p 0, presence_penalty 1.5. Saved model defaults still retain the owner's
general-task preset (temperature 1, repetition_penalty 1).

Both engine configs use 262144 context capacity, Q8 KV, CPU vision, window 8,
lookup 3, tail skip 0, fast attention/decode and 1024 MiB VRAM reserve. IQ3 retains
CPU IQ reuse. Q4 starts with a new expert profile; IQ3 inherits an older profile.
No compiler, downloader or hash job ran during these requests.

| Updated Poly engine | Output tokens | Decode tok/s | First engine token | Draft accepted/offered | Expert cache hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Atomic Uncensored IQ3 | 559 | 35.2 | 1.562 s | 198/301 | 57.3% |
| Bartowski Q4, first request | 770 | 23.0 | 5.957 s | 274/387 | 41.1% |
| Bartowski Q4, repeat | 766 | 25.8 | 0.239 s | 297/411 | 43.4% |

The repeat reused 91/92 prompt tokens, so its first-token timing is not a cold
prefill comparison. All reported TPS values are actual engine decode metrics
(`projection: null`), not total wall-clock output speed. Raw sanitized metrics
are in the three JSON files. IQ3's prior unmodified-engine UI run was 36.5 tok/s
(402 tokens); stochastic output, cache history and sample count prevent a
strong regression/performance conclusion.

Q4 produced coherent Turkish pipeline descriptions, including wake word/VAD,
STT, actions, TTS, streaming and interruption handling. Its first illustrative
Python snippet is not executable production code, and recommendations were not
validated. This establishes basic functioning, not superior coding quality.
Bartowski is regular Ornith, Atomic is Uncensored: checkpoint variant and quant
both differ. No coding evaluation, tool-call test, image request, full 256K
request, multi-GPU run, or long-context memory-pressure test was performed.

## Deployment

The final image uses app version `0.2.2+poly.mixed-gu`; Q4 is the compose boot
default and is left loaded for user testing. The inference port stays internal
to the container, only the app's loopback port is published. LiteLLM, Pi and
router wiring remain unchanged. GPU 0 workloads were not modified. Atomic
IQ3/IQ4 files and configs remain available.
