# Optional PolyStrata engines in this fork

This branch starts from Niko1221/Strata `011c9e60` (2026-10-21). It keeps the original app, Control, CLI,
setup flow and native Qwen engine. It imports the Poly engines from ersensari/PolyStrata `f6c93f63`, including
the GQA16 attention kernels and opt-in CPU IQ multi-token reuse from `f5cfcd4a`. The existing MIT license
applies; the imported sources retain their upstream comments and third-party attributions.

No upstream pull request is planned. This is a source integration branch, not a published desktop installer
or a migration of existing application routing. A separate [local container trial](../docker/local-poly/README.md)
now runs the original web app host with the imported engines.

An opt-in [Ornith native-engine experiment](ORNITH_NATIVE.md) also runs the existing IQ3 GGUF through
the original `strata` executable. It is separate from the default Poly deployment and currently slower
on the tested 16 GB host.

## Scope

- `src/poly/` and `include/strata/poly/`: Qwen35 MoE (Ornith/MiniCPM), Gemma4, GLM5-next, DeepSeek4 and LFM2 MoE.
- `cmake/poly_engines.cmake`: optional `strata-poly` executable and its unit-test targets.
- Poly's required versions of native MMVQ/IQ CUDA kernels have their own `strata::poly_kernels` namespace.
  The original Qwen CUDA source files and headers are unchanged.
- The Python server gains Poly-specific tokenizer metadata, prompt/tool formats and image markers. Original
  Qwen tool conversion and reasoning-effort precedence stay the default. Unknown Poly tool calls remain text
  when they do not name a declared tool.
- `tools/poly_import.py` registers **existing local** GGUF files. It neither downloads models nor starts an engine.
  It refuses missing shards and existing config/pack paths. Generated API keys are not printed; configs are private files.

The app's upstream model catalog/setup wizard is unchanged. It does not acquire a new Poly model downloader.
Use an explicit imported run config with this branch's source server. The upstream CLI's normal installation
and update commands still fetch upstream components, which do not contain `strata-poly`.

## Build (Linux/CUDA)

Use Strata's normal compiler, CUDA and Python dependencies. The engine's llama.cpp/ggml dependency stays at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, the pin already used by this base. Point `STRATA_GGML_DIR` at
that checkout. For the tested RTX 5060 Ti / CUDA 13 machine:

```sh
cmake -S . -B build-poly -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_POLY_ENGINES=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSTRATA_GGML_DIR=/absolute/path/to/llama.cpp-3cf03257
cmake --build build-poly --target strata-poly strata -j 8
```

`STRATA_POLY_ENGINES` defaults to `OFF`. HIP code is retained from Poly, but this integration has only been
built and exercised with CUDA. Metal, SYCL and wave64 HIP are not Poly build targets here.

Poly's MiniCPM/Gemma vision support needs the newer, **separate encoder-only** llama.cpp checkout
`69f201a2051ea9b9e9b50c3cc56afd9e2ae64414`. Do not substitute this checkout for the engine's ggml dependency:

```sh
cmake -S tools/vision -B build-poly-vision -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE=OFF \
  -DLLAMA_DIR=/absolute/path/to/llama.cpp-69f201a
cmake --build build-poly-vision --target strata-vision -j 8
```

This builds a CPU encoder, keeping VRAM for the LLM. Native CPU builds are for this machine; keep portable
build settings when distributing binaries to another CPU.

## Register and start a local model

For a split model, pass the **first shard**. Use host GPU 1 in this homelab; use `--gpu 0` inside a container
that exposes only host GPU 1. Never expose or select the unrelated GPU 0 workloads for these tests.

```sh
python tools/poly_import.py \
  --model /absolute/path/to/first-shard.gguf \
  --engine "$PWD/build-poly/strata-poly" \
  --name ornith-iq3 --config-dir /absolute/path/to/private-poly-configs \
  --context 262144 --threads 12 --gpu 1 \
  --fast-attention --iq-multi-token --dry-run
```

Remove `--dry-run` to extract the tokenizer and write the config. To enable images, also pass both
`--mmproj /absolute/path/to/mmproj.gguf` and `--vision-exe "$PWD/build-poly-vision/strata-vision"`.
The importer sets library search paths for the engine and encoder directories. Keep any required shared
libraries there, or add their directories to the config's `lib_dirs`.

The importer selects Q8 KV for Qwen35 models, reserves 1024 MiB VRAM, and uses coding sampling defaults
(`temperature=.6`, `top_p=.95`, `top_k=20`, `min_p=0`, `presence_penalty=0`, `repetition_penalty=1`). Context
is a capacity setting, not evidence of a completed 256K quality or memory test. CPU IQ reuse and Poly flash
attention are explicit opt-ins, not automatic changes to the native Qwen engine. CPU IQ reuse changes the
order of floating-point summation; generated tokens need not be identical with it on and off.

**Stop the current model service on the same GPU before launching this server.** Do not run the production
PolyStrata model beside it. The following command starts the real engine; omitting `--engine strata` starts
the upstream mock server instead:

```sh
python -m serve.server --engine strata --port 8080 \
  --config /absolute/path/to/private-poly-configs/strata-ornith-iq3.json
```

This binds to loopback by default and serves the original Strata web interface and API. It is separate from
the Tauri desktop app, which was not packaged/deployed during this integration.

## Validation

See [the integration record](../bench/results/2026-10-21-poly-port/README.md) for the exact checks and limits.
The short model checks establish that the new server and engines communicate correctly. They are not
comparative throughput benchmarks, coding evaluations or long-context acceptance tests.
