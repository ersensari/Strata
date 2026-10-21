# Local homelab container

`forgeloop-ai-strata` serves the original Strata 0.2.2 app host and this fork's Poly/native engines.
UI: **http://127.0.0.1:10110**. Only host GPU 1 is exposed; it appears as GPU 0 to the engine.
Startup loads **AtomicChat Ornith IQ3_XXS**, 262144 context capacity, Q8 KV, 12 threads, one request at a time,
1024 MiB reserve, CPU vision and fast attention enabled (CPU IQ reuse remains enabled for the Atomic IQ3/IQ4 configs).

As of 2026-10-22 this is the live fork. The deployment overlay in the homelab repository,
`/home/ersen/Source/forgeloop-ai/docker/strata/compose.strata.yml`, connects it to the shared
network and native-aware model router. Provider traffic goes through that router. Both
http://127.0.0.1:10104 and http://127.0.0.1:10110 serve the original Strata UI and model controls.
The inference worker itself stays on container loopback. Existing PolyStrata and KT stay stopped.

The separate named volume `forgeloop-ai-strata-local-data` stores configs, profiles, chats and workspace.
The existing `forgeloop-ai_forgeloop-ai-polystrata-data` volume is mounted **read-only** for prepared packs.
All model directories are also read-only. Retain that legacy volume: this container depends on its packs.
Six configs are registered: Ornith IQ3, Ornith IQ4, MiniCPM, uncensored MiniCPM, Gemma and Qwen.
Only Ornith IQ3 loads at boot. The UI or provider router switches the single loaded model.
Qwen uses this fork's original native engine; the other models use its optional Poly engine.
The Bartowski Q4_K_M experiment is retired: its two weight shards and dedicated mmproj were deleted
at the owner's request (79,830,461,312 bytes), its model config is removed, and it is no longer mounted.
The shared AtomicChat IQ3/IQ4 mmproj and all their shards are preserved.

## Prepare and run

This local image reuses the existing `forgeloop-ai-polystrata:0.1.45-poly.2` CUDA/Python runtime. It is not
a self-contained public build. Build the engines and CPU encoder as described in
[POLY_ENGINES.md](../../docs/POLY_ENGINES.md), then copy `strata-poly`, `strata` and `strata-vision` into
`docker/local-poly/artifacts/` (ignored by Git). Build the host with `npm ci && npm run build` in
`apps/host/`. The image adds pinned Node 24.14.0 and copies the app and local binaries.

For the full live deployment, use the homelab overlay README. For this container alone, from the Strata repository root:

```sh
docker stop forgeloop-ai-polystrata
COMPOSE_IGNORE_ORPHANS=true docker compose --env-file /home/ersen/Source/forgeloop-ai/.env -p forgeloop-ai \
  -f docker/local-poly/compose.yml up -d --build
docker logs --tail 30 forgeloop-ai-strata
```

`COMPOSE_IGNORE_ORPHANS` only suppresses warnings about this homelab's other services. Do **not** use
`--remove-orphans` or `down -v`; those services/data belong to the same wider Compose project.
The entrypoint refreshes its source copy, preserves saved configs, starts the original Node host, and
asks its native controller to load exactly the configured default model. No setup downloader runs.
The original app then owns the inference process, including unload and model switching.

Stopping this service:

```sh
docker stop forgeloop-ai-strata
```

Leave it stopped before restoring old PolyStrata. The old service's own startup default remains MiniCPM;
starting it again will load that model unless its separate default is deliberately changed.

## Verification (2026-10-21)

- Container running/healthy; native controller reports Ornith IQ3 ready at internal port 8080.
- Browser displays the new app with the loaded Ornith model. A real UI request, `Yalnızca Hazırım yaz.`,
  returned `Hazırım`. This was a functionality check, not a speed benchmark.
- Old production and development containers are stopped. Other applications' routing is unchanged.
- Fixed a source-app startup incompatibility: the native controller used to pass `STRATA_API_KEY=''`,
  which the current server explicitly rejects. It now removes that environment override, so each private
  run config supplies its own key. The helper's regression test and native control/chat tests pass.

## Bartowski Q4_K_M trial (2026-10-21)

Downloaded `bartowski/Ornith-1.5-122B-GGUF` Q4_K_M (two shards, 78.92 GB) and its f16 mmproj
to `/mnt/llm-models/ornith-1.5-122b-bartowski`. Both shards and mmproj passed SHA256 validation
against the pinned Hugging Face revision. The directory is mounted read-only and a separate config
is registered with 262144 context capacity, Q8 KV and CPU vision. No model files were re-quantized.

**This model did not load.** Thirteen layers have different gate and up expert quant types, including
Q4_K gate / Q6_K up in block 0. The current Poly engine combines gate/up under one type and rejects
this layout. Startup was restored to IQ3; Q4 TPS and output quality were not measured. The downloaded
model and separate config remain available for future mixed-type support. Do not select it expecting
a working model yet. This is an engine limitation, not a failed download or a CUDA OOM.

The same-day IQ3 comparison through the new app, thinking off, returned 402 tokens at 36.5 tok/s.
See [the trial record](../../bench/results/2026-10-21-bartowski-q4/README.md) for settings and limitations.

## Mixed gate/up support (2026-10-22)

The earlier Q4 rejection above is historical. The Poly engine now keeps separate gate/up types,
strides and sizes, including GPU-cache allocation, refill and CPU activation quantization. Existing
same-type models retain their combined gate/up products. No model weights were converted or re-quantized.

Real Q4 CPU/GPU expert checks and IQ3 regression checks passed. In the original app with thinking
off, the voice-assistant question produced 23.0 tok/s cold and 25.8 tok/s on a repeat; IQ3 on the
updated engine produced 35.2 tok/s. These short runs do not establish quality or long-context performance.
Q4 is the boot default and is left loaded for user testing, still standalone on host GPU 1.

Mixed-type layers currently use the small-group path during prompt prefill as well as decoding;
the single-quantizer tiled prompt and GPU-only-routing paths are bypassed for those layers.
Long prompts can therefore be slower. CUDA was tested; HIP was not available for validation.
See [the support record](../../bench/results/2026-10-22-poly-mixed-experts/README.md).

## Live migration (2026-10-22)

The startup helper also imports the previously omitted uncensored MiniCPM config.
It configures the app gateway's existing API-key setting from STRATA_API_KEY,
without changing individual engine secrets. Existing saved configs, profiles and chats
are preserved. The homelab router translates the provider aliases to these installed
model IDs and polls the original app's native load jobs. Model switching and chat
continue to belong to the original app. Manage models now offers Load for installed local
configs outside the download catalog, using the saved config without applying settings
from the unrelated catalog selection. No upstream PR was opened.

Historical Q4 trial notes above describe the earlier experiment, not the active model.
