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

Initial settings:

- Physical GPU 1, selected by Strata's existing child environment.
- 32768 context, Q8_0 K/V, one request slot, 12 CPU threads.
- Target GPU layers enabled; the first 20 MoE layers keep experts on CPU.
- DSpark on GPU, `--spec-type draft-dspark --spec-draft-n-max 15`.
- Vision projector on CPU to leave GPU room for target verification and draft.

These are initial settings, **not measured optimal placement**. Fifteen is
the maximum draft length, not a promise of fifteen accepted tokens per step.
Full 256K context and this placement have not been validated on GPU.

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

The active Ornith worker and host PID were preserved throughout preparation.
**The real MiniCPM + DSpark pair has not yet been loaded or benchmarked.** Next
checks are GPU memory, a real text request, image and tool requests, then matched
draft-on/off timing and acceptance measurements before choosing a default.

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
