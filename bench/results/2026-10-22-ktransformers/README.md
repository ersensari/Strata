# Ornith Q4_K_M on KTransformers, 2026-10-22

Local experiment requested by the owner. Strata was stopped before starting
KTransformers. No router, Pi, LiteLLM or other consumer was connected. No PR.

## Configuration

- Ryzen 9 9900X, 12 physical cores / 24 threads, one NUMA node, AVX-512 BF16;
  no Intel AMX. CPUInfer: 12 threads, one pool, LLAMAFILE.
- Host GPU 1 only: RTX 5060 Ti, 16311 MiB. GPU 0 workloads untouched.
- kt-kernel 0.7.2, sglang-kt 0.7.2, sgl-kernel 0.3.21,
  Torch 2.9.1+cu128, transformers-kt 5.6.0.post6; full freeze attached.
- Reviewed KTransformers commit a694e718e8b3ec2e221277d674e369f3be38d2a4;
  its SGLang submodule points at 541ddc37cbc92c60dc748db5ff1a2aad0b069a80.
  Runtime uses published wheels, not a source build of those commits.
- Routed CPU experts: existing Bartowski Ornith-1.5-122B Q4_K_M (78.92 GB).
  Source/hash provenance is beside the model files and in the preceding mixed
  expert benchmark record. No re-quantization or modifications to those files.
- GPU: original Ornith BF16 attention, shared experts, router, embedding/head,
  and vision tensors. 1068 tensors, 13,197,401,568 raw bytes. Source revision
  is in model-source.json. Downloaded selected HTTP byte ranges, reconstructed
  safetensors headers/index, excluded routed experts and MTP. Checkpoint bounds
  and index correspondence checked (checkpoint-check.json). This is not the
  full upstream BF16 checkpoint and is specific to this KT loading path.
- 8K context, BF16 KV, one request, 256-token chunked prefill, no MTP,
  zero GPU routed experts, no deferred experts. GPU load: 12.42 GiB weights;
  total device allocation about 14.0 GiB during requests. Graph capture: ~80 MiB.
- Text inference only tested. Vision weights are present because SGLang's
  `--language-only` is an encoder-disaggregation mode requiring encoder URLs,
  not a switch that removes the vision module. Vision requests not tested.

## Validation and setup findings

CPU numerical check uses real layer-0 Q4_K gate / Q6_K up GGUF weights and
compares KT with independent GGUF dequantization + float32 PyTorch SiLU/MatMul.
Relative L2 error: 0.0134146; finite output; clean destruction. 100 iterations
of fixed experts measured 0.405 ms/layer, **not whole-model TPS** and not
representative of changing routed experts or 48 layers.

The first CPU-only asynchronous test returned premature zero output and failed
its numerical assertion; another callback-based CPU test segfaulted at exit.
The passing CPU-only check uses the project's `KT_FORCE_SYNC_SUBMIT=1` escape
hatch. The GPU server uses the normal asynchronous path, without that override;
real full-model requests then passed. SYS_NICE was needed for NUMA binding.

The SGLang wheel alone lacked the common_ops binary; installing sgl-kernel
fixed that. libnuma1 and Python 3.12 headers were needed. SGLang rejected Torch
2.9.1 with cuDNN 9.10, so cuDNN was upgraded to 9.16.0.29 as its diagnostic
recommended. That override conflicts with Torch's strict package metadata pin.
The server passed its own cuDNN check and generated real responses afterward.
No SGLang/KT source patches or model architecture aliases were added.

## Measurement

Prompt, without a system message:

> bir sesli asistan yapsaydin mimariyi nasil kurardin

28 prompt tokens, cap 768 output tokens, seed 42, thinking disabled. Every run
hit the output cap, so these are truncated answers. Sampling and every SSE event
are archived in the JSON/JSONL files. Text files trim trailing whitespace; the
JSON content preserves the original output.
Decode TPS = (completion_tokens - 1) / (last content time - first content time),
measured by the streaming client. First requests after a restart include JIT
compilation/warmup effects; use the second requests for a warm observation.
Repeated prompts can benefit from prefix/file caches. This is not a controlled
cold-cache comparison or a statistically powered engine benchmark.

| Run | Output tokens | First content (s) | Total (s) | Decode TPS |
|---|---:|---:|---:|---:|
| trial-1 | 768 | 36.313 | 100.737 | 11.91 |
| trial-2 | 768 | 0.439 | 57.052 | 13.55 |
| graph-1 | 768 | 43.581 | 110.279 | 11.50 |
| graph-2 | 768 | 0.850 | 55.524 | 14.03 |

The warm graph run is the final tested deployment setting. This initial KT
configuration is slower than the earlier Strata Q4 UI observations (~23-26 TPS),
but settings differ: Strata had GPU expert caching, MTP, quantized dense weights,
Q8 KV and a 256K allocation. This is not evidence that every KT configuration is
slower. No AMX, GPU expert cache, GPU quantization, MTP, long-context performance,
coding benchmark or vision quality was evaluated here.

Outputs are coherent Turkish architecture outlines (audio capture, STT, dialog,
tools, TTS). They contain occasional awkward phrasing and illustrative code that
is not executable as written. This single truncated answer is not a quality
benchmark. CUDA Graph did not produce a large speedup in these observations.

The final KT API is left running at http://127.0.0.1:10112. Strata remains stopped.
Deployment and setup caveats are in docker/local-kt/README.md. Fetch scripts are
archived local preparation helpers; the header helper is pinned to the observed
revision and fetch-vision resolves its sibling script relative to itself.
