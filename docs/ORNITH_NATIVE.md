# Experimental Ornith 1.5 on the original Strata engine

The original CUDA `strata` executable already dispatches GGUF architecture `qwen35moe` to its own
`src/qwen35/q36_engine.cu` driver. Ornith's 31-shard AtomicChat AD-IQ3_XXS file passes this driver's
geometry/layout validation: 48 trunk layers + one MTP block, hidden 3072, FF 1024, 256 experts/top-8,
32 attention heads/2 KV heads, head dimension 256, SSM state 128, conv 4. All 899 tensors were checked.

The actual missing pieces were quant support and CPU scratch capacity. This fork adds the opt-in
`STRATA_ORNITH_NATIVE` build option (default OFF):

- Add IQ3_S and Q4_K down projections to native grouped GPU expert kernels. The real file uses
  IQ3_XXS/IQ3_S in the trunk and Q4_K/Q4_K in the MTP block.
- Increase native CPU down-activation scratch to 2048 bytes for this build; FF 1024 needs 1168 bytes
  when quantized as Q8_K. The default build keeps its original capacity.
- Add `ornith_native_parity`, the existing independent ggml float/CPU/GPU expert check compiled at
  Ornith's actual 3072/1024 dimensions. The normal test dimensions remain 2560/640.

No GGUF metadata or weights are rewritten. A user-facing profile may be named `ornith15moe`, but the
file's actual architecture stays `qwen35moe`; a new architecture identifier alone adds no MTP capability.

## Build and initial run

Add `-DSTRATA_ORNITH_NATIVE=ON` to the CUDA configuration in [POLY_ENGINES.md](POLY_ENGINES.md), then:

```sh
cmake --build build-poly --target strata ornith_native_parity native_expert_parity -j 8
build-poly/ornith_native_parity /path/to/first-ornith-shard.gguf 0 47 48
build-poly/native_expert_parity --synthetic iq3_xxs/iq4_nl q4_K/q5_1
```

The following **engine-only** initialization check reads stdin to EOF after loading; it does not start a
web server. Stop every other model process on the same GPU first. Set GPU visibility for your environment
(only host GPU 1 is allowed in this homelab; CDI exposes it as device 0 inside the test container).

```sh
STRATA_Q8_PACKED=0 build-poly/strata --serve \
  --model /path/to/first-ornith-shard.gguf \
  --max-context 32768 --chunk 512 --cpu-threads 11 --mtp 3 --expert-vram 6.4 </dev/null
```

For the Python source server, use these arguments in a **separate private run config**, set `exe` to the
new `strata` binary, `arch` to `qwen35moe`, and reuse the original tokenizer directory. Set
`env: {"STRATA_Q8_PACKED": "0"}` and include the binary directory in `lib_dirs`. Keep `libstrata_kq.so`
beside `strata` for the prompt MMQ path. Do not copy the Poly `--kv q8_0`, `--window`, `--lookup` or
conversation-cache flags into a native config: this driver's supported flags/behavior differ.

## Important limits in this original driver

- KV caches are FP32 and allocated for the configured capacity. There is no Q8 KV flag in this path.
  The experiment used **32K**, not a 256K acceptance test.
- Dense Q8 packed decode copies are on by default. On this 16 GB card the initial native load failed for
  lack of hybrid-buffer space. Disabling the optional copies retained the weight values while reducing
  loaded dense-weight allocations from 10.18 GiB to 6.43 GiB.
- With the tested buffers/staging, only 12 experts per layer occupied VRAM (2.09 GiB); 44.69 GiB of expert
  weights were copied into pinned host memory. Initial setup took roughly 55–59 seconds.
- MTP exists already (`--mtp 3`). This driver's serve loop enables it only for **greedy requests without
  penalties**: `temperature=0`, `presence_penalty=0`, `frequency_penalty=0`, `repetition_penalty=1`.
  Nonzero temperature or penalties selects the one-token sampling path. These restrictions were **not**
  relaxed: doing that correctly needs target-distribution verification, not just removing a guard.
- The native and Poly drivers perform different computations/activation rounding and cache placement.
  Text identity between the two is not an acceptance criterion here; no coding-quality equivalence is claimed.

## Result

Ornith successfully generates text on the original native engine, including verified MTP drafts.
This does **not** currently provide a speed improvement over the installed Poly engine on this host.
The normal UI deployment stays on Poly IQ3 with Q8 KV and 256K capacity. The experimental binary/config
was not connected to LiteLLM, Pi, the router or another application.

See [the experiment record](../bench/results/2026-10-21-ornith-native/README.md) for timings and validation.
