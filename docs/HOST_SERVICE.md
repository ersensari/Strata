# Host service for this fork

This homelab deployment runs the source checkout at `/home/ersen/Source/Strata`
(branch `codex/poly-engines`) under the user's systemd manager. Only host GPU 1
is selected in every saved model config. Keep the Docker Strata, old PolyStrata
and KTransformers services stopped before starting it. No upstream PR.

## Install and runtime

Follow [INSTALL.md, Linux source install](INSTALL.md#from-source-linux).
`./setup.sh --help` initializes the checkout's private `.venv`; runtime packages
come from `requirements.txt`, with setup.py's pinned CUDA 13 runtime/cuBLAS wheels.
No global Python or driver change is needed. `./setup.sh --check --gpu 1` checks
this machine without acquiring a model or starting inference.

This migration reuses the fork's already built native 0.1.47 engine in `engine/`
and its custom Poly engine in `build-poly-host/`. The native BUILD.json records
its actual local binary version and source hashes. Poly has its separate CPU
vision encoder; the native engine and encoder remain independently updateable.
All dynamic CUDA libraries are in `.venv/lib/python*/site-packages/nvidia/cu13/lib`.

The six private `strata-*.json` configs live at the checkout root (git ignored).
Their absolute Docker paths were translated to host paths, their GPU changed
from container index 0 to physical index 1, and context, KV, MTP and sampling
settings were preserved. Raw GGUF weights stay in `/mnt/llm-models`. Only required
prepared packs, tokenizers and the active MTP subset were copied to
`/mnt/llm-models/strata-runtime` (about 2.3 GB). `packs` and `mtp` in the checkout
are symlinks to that runtime folder. Expert routing profiles were adopted with the model settings (an empty profile is invalid for the native engine). No chat history or request logs were imported.

Host state is `/home/ersen/.local/share/strata-fork`: fresh `ui/`, `workspace/`,
`profiles/`, `logs/`, and the source install's `settings.json`.
`~/.config/strata-fork/service.env` is mode 600 and contains the existing serving
key and the CUDA library path. Never commit model configs or this environment file.

Copy `deploy/systemd/forgeloop-ai-strata*` to `~/.config/systemd/user/`, adapt paths
and Node version if necessary, then run:

```sh
systemctl --user daemon-reload
systemctl --user enable --now forgeloop-ai-strata.service forgeloop-ai-strata-access.socket
systemctl --user status forgeloop-ai-strata.service
journalctl --user -u forgeloop-ai-strata.service -n 40
```

This account already has lingering enabled, so its enabled user units run without
an interactive login. The service uses Node 24.14.0, Type=notify, a private umask,
and systemd cgroup cleanup. It announces ready only after the original app host
loads its default Ornith IQ3 config. Startup never runs a model downloader. The
original native controller owns model switching and engine children.

## Connections

- Web UI and authenticated `/v1`: `http://127.0.0.1:10104`.
- Compatibility URL: `http://127.0.0.1:10110`.
- `forgeloop-ai-strata-access.socket` also listens on Docker's host gateway
  `172.17.0.1:10110`, forwarding to the same loopback app via systemd-socket-proxyd.
  Adapt the gateway address if Docker's default bridge changes. There is no
  wildcard/LAN bind.
- The existing authenticated router stays in Docker on loopback 10105; its
  `UPSTREAM` is `http://host.docker.internal:10110` with `host-gateway`.
- The provider stays in Docker on 10080, using the same router and model aliases.

The deployment overlay and controller registry are maintained in
`/home/ersen/Source/forgeloop-ai/docker/strata/`. The old Strata container is gated
by its explicit `strata-container` profile; ordinary compose runs must not start it.
The controller's read-only Strata row observes the router and its upstream health.

## Updates and model controls

Qwen's native engine status comes from `engine/BUILD.json`. Native source-mode
engine update/rollback controls remain the original Strata controls and require
all models unloaded. In a source install `setup.py --update` follows the version
required by the checked-out source; a local engine is rebuilt when its engine
sources change. A compiler/CUDA Toolkit is needed if a future source change
requires a rebuild. This is distinct from the packaged app's signed component feed.

Custom `strata-poly` and its encoder are maintained separately using
[POLY_ENGINES.md](POLY_ENGINES.md); an upstream native engine update does not replace
them. The app's automatic official-repository source update refuses this fork by
design. Merge upstream main manually when the owner requests it, preserve the fork
patches, rebuild affected engines and the web UI, then restart the service. Do not
point this fork at an upstream packaged Control bundle, which lacks the Poly path.

## Stop or roll back

```sh
systemctl --user stop forgeloop-ai-strata-access.socket forgeloop-ai-strata-access.service forgeloop-ai-strata.service
```

To return to Docker, stop and disable those host units first. Restore the router
UPSTREAM to `http://forgeloop-ai-strata:8791`, explicitly start the old container
through its `strata-container` profile, then recreate only the router. Old volumes
remain available for rollback. Do not start both deployments on GPU 1. Never use
`--remove-orphans` or `down -v` on the shared homelab Compose project.

## Verification, 2026-10-22

The source installer PC check, shared-library resolution and systemd unit
validation passed. Native status reports local CUDA engine 0.1.47 and all six
adopted configs ready. Seven authenticated provider calls exercised every model
(including native Swift, both Ornith quants and both MiniCPM variants), returned
HAZIR and confirmed physical GPU index 1. IQ3 was restored afterwards. These are
functionality checks, not long-context, vision-quality or speed benchmarks.
Sanitized results are in `bench/results/2026-10-22-host-service/`.

The source UI also preserves each unloaded model's saved GPU choice instead of
preselecting both detected cards. GPU labels in the UI are one-based: its GPU 2
is nvidia-smi's physical GPU index 1 used by this service.

An explicit `update-engine` job also completed after unloading the model. SHA-256
checks confirmed both the current native binary and the separate Poly binary
were unchanged; this verified the source-mode update path at the currently
required version, not acquisition of a newer release.

The opt-in `STRATA_SOURCE_ENGINE_UPDATES=1` in this service also enables an honest
source-mode check in Settings → About: the target is the version this checkout's
installer can supply, and a changed native source hash offers a rebuild through
the original update action. "Current" means current for this checkout, not that
all upstream commits have been merged. Other deployments keep the upstream
update-check behavior unless this option is explicitly enabled.
