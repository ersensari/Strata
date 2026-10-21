# Local Ornith KTransformers experiment

This is an opt-in experiment, separate from Strata. Do not run it while Strata
holds GPU 1. It exposes only `127.0.0.1:10112`; it is not connected to the router.
There is no chat UI in this deployment. The OpenAI model name is `ornith-kt-test`.

Hardware: Ryzen 9 9900X (12 physical cores, AVX-512 BF16, no AMX), 96 GB RAM,
host GPU 1 RTX 5060 Ti 16 GB. Only that card is exposed as CUDA device 0.

## Existing local installation

From the Strata checkout:

```sh
COMPOSE_IGNORE_ORPHANS=true docker compose -p forgeloop-ai -f docker/local-kt/compose.yml up -d --no-build
curl -f http://127.0.0.1:10112/health
```

Stop this service before restoring Strata:

```sh
docker compose -p forgeloop-ai -f docker/local-kt/compose.yml stop
```

Do not use `down --remove-orphans`: the Compose project is shared with the lab.
The external volume contains the Python environment and caches. The external
network and volume were created with `forgeloop.managed=true`,
`forgeloop.project=forgeloop-ai`, and `forgeloop.resource` ownership labels.

## Preparation on a fresh machine

The Dockerfile extends the existing local Strata CUDA development image. It adds
libnuma and Python headers; the Python packages live in the external volume,
not the image. This is deliberately a local experiment, not a standalone image.
Create the labeled external network/volume first, then build this image.
Use `compose.yml` plus `setup.override.yml` to run an idle setup container.
Inside it, prepare `/data/venv` with Python 3.12 and install:

```sh
python3 -m venv /data/venv
/data/venv/bin/pip install --cache-dir /data/pip-cache kt-kernel==0.7.2 sglang-kt==0.7.2 sgl-kernel==0.3.21
/data/venv/bin/pip install --cache-dir /data/pip-cache nvidia-cudnn-cu12==9.16.0.29
```

Set `TMPDIR=/data/tmp` (create it first) so large installation temporary files
use SSD storage. The cuDNN override is required by SGLang's Torch 2.9.1 Conv3d
compatibility check. It deliberately differs from Torch's strict metadata pin;
`pip check` reports that conflict. Tested installed versions are archived with
the benchmark results. The package is `sglang-kt`; its internal version string
can still say `0.0.0.dev0`.

Model mounts are read-only. Existing Bartowski Q4_K_M GGUF shards provide routed
CPU experts. The GPU checkpoint is a selected subset of original BF16 HF
safetensors (attention, shared experts, router, embedding/head, vision), without
routed experts or MTP. Preparation scripts and source revision are in
`bench/results/2026-10-22-ktransformers/`. No full 250 GB BF16 download is needed.
Run fetch-headers, fetch-dense, then fetch-vision in that order. They target the
local paths in compose.yml. HTTP ranges must be honored by the source server.

SYS_NICE enables the kernel's NUMA binding. No privileged mode is used.
The experiment starts at 8K context, one sequence, BF16 KV, no MTP, no GPU routed
experts. It does not reproduce Strata's 256K/Q8/MTP/cache configuration.
