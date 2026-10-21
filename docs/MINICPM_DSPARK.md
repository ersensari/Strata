# MiniCPM with DSpark (local, experimental)

This fork can run MiniCPM-V 4.7 35B-A3B with its DSpark GGUF through an
optional, owned llama.cpp worker. This does not add DFlash kernels to Poly.
The ordinary native Qwen and Poly configurations keep their existing paths.

Select **MiniCPM-V 4.7 35B A3B · Q4_K_M + DSpark** in the app. The local
configuration is `strata-minicpm-dspark.json`; it is private and not committed.
The ordinary MiniCPM profile stays available for comparison.

## Local preparation, 2026-10-22

The target is the corrected 40-layer MiniCPM Q4_K_M GGUF. The vision projector
and `dspark-MiniCPM-V-4_7-35B-A3B-Q8_0.gguf` already exist alongside it under
`/mnt/llm-models/Strata-data/models/minicpm-v-4.7-35b-a3b`.

The local llama.cpp b11552 executable and libraries were extracted from the
already installed official CUDA server image, into
`~/.local/share/strata-fork/engines/llama-dspark`. The image identity is recorded
in that folder's `BUILD.json`. No new model downloads or background Docker
services are required.

Current local settings (updated after the first GPU load):

- Physical GPU 1, selected by Strata's existing child environment.
- 262144 context, Q8_0 K/V for target and draft, one request slot, 12 CPU threads.
- Target GPU layers enabled; the first 32 MoE layers keep experts on CPU.
- DSpark on GPU, `--spec-type draft-dspark --spec-draft-n-max 15`.
- Vision projector on CPU to leave GPU room for target verification and draft.

These are initial settings, **not measured optimal placement**. Fifteen is
the maximum draft length, not a promise of fifteen accepted tokens per step.
The original profile used 32K context and 20 CPU expert layers. At 256K the
20-layer placement exhausted GPU memory before the draft loaded; 32 CPU expert
layers loaded successfully with about 12.5 GiB GPU usage. Context allocation
passed, but generation through a full 256K conversation has not been tested.

The first real load exposed a missing dependency in the locally extracted
runtime: `libggml-cuda.so` needs `libnccl.so.2`, although `llama-server` itself
passes ldd without it. llama.cpp silently fell back to CPU. NCCL and the CUDA 12
runtime/cuBLAS libraries are now copied from the same pinned container image
into the engine folder. Validate the CUDA plugin as well as the main binary.
Both local llama profiles now require a successful CUDA device probe before
loading, refusing silent CPU fallback.

## Ownership and API

`serve/llama_backend.py` starts llama-server as a child of the existing owned
Strata worker. Strata keeps its authenticated public API, instance identity,
owner lifeline and readiness attestation. The child binds a random loopback
port with a separate temporary API key. Unloading or stopping the Strata worker
terminates the child; restarting the whole host service is unnecessary.

OpenAI chat/completions, Anthropic messages/counting, tokenization and template
requests go directly to llama.cpp after Strata authentication. Image blocks,
tools and sampling fields are forwarded intact. Streaming is progressive, and
disconnect/drain cancellation closes the upstream socket. llama.cpp performs
the model's actual chat templating, vision processing and DSpark verification.

Context and the shared draft/picture switches apply on the next load. Turning
draft off removes the DSpark arguments, allowing a same-engine baseline.
Poly-specific engine arguments and tuning do not control this child; edit the
private profile's `llama_backend.args` for its llama.cpp placement/KV settings.
Strata-managed MCP execution is deliberately rejected on this backend; clients
can still execute model tool calls themselves.

Strata's token-pipe request monitor and token-rate counters do not account for
forwarded inference. This profile disables the API monitor. Read llama.cpp
timings in the response/log, or its authenticated `/llama-metrics` route, for
actual performance. Do not interpret Strata's absent token counters as zero TPS.

## Validation and remaining checks

Preparation passed local dynamic-library resolution and actual b11552 argument
parsing, without exposing a GPU. HTTP tests cover authentication, origin policy,
image/tool/sampling pass-through, SSE, disconnect cleanup, upstream failure,
draining, and draft/context switches. A separate CPU-only fake-child smoke test
exercised the real `serve.server` entry point, ready-file publication, telemetry,
chat forwarding and shutdown.

Ornith was preserved during initial preparation. The later CPU-fallback fix
required restarting the host to refresh settings validation and reloading the
DSpark worker. The live settings-file check now reports no issues for
`display_name` or `llama_backend`. The real MiniCPM + DSpark pair loaded on GPU 1
with 256K allocated, and a short provider request returned `OK` (17 prompt and
2 completion tokens). This is a smoke test, not a speed benchmark. Real image
and tool requests, long-context generation, and matched draft-on/off timing
and acceptance measurements remain necessary before choosing a default.

References: [MiniCPM DSpark model card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf),
[llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).

## Swift 27B dense profile

The same owned worker can also run the existing local Swift 1.5 Qwen 27B
IQ3_S GGUF (`qwen35`, dense, 64 target layers plus one embedded MTP block).
This is distinct from the native Qwen Flash Next MoE model.
`strata-swift-27b.json` registers **Swift 1.5 Qwen 27B Dense · IQ3_S + MTP**.
The 12,120,016,896-byte model stays in its existing `swift-1.5-27b` directory;
only its tokenizer is extracted into Strata-data. No second model copy is made.

Initial settings are GPU 1, 32K context, Q8_0 K/V, `draft-mtp` with at most three
draft tokens and GPU layer offload requested. There is no mmproj in this local
installation, so the profile is text-only. Header bounds, MTP tensors, argument
parsing and registration were checked without interrupting Ornith. Actual GPU
fit, MTP acceptance, throughput and long-context behavior remain unmeasured.
