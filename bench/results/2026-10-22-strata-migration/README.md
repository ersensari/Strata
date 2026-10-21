# Live original Strata fork migration, 2026-10-22

The owner requested retiring KT, restoring the original Strata main fork, migrating
all former PolyStrata models and provider/model UI traffic, enabling Ornith IQ3/IQ4,
and deleting the Bartowski Ornith Q4_K_M experiment. No PR.

- KT and old PolyStrata stay stopped. Only host GPU 1 is exposed to the new Strata.
- The homelab overlay docker/strata/compose.strata.yml deploys this fork's local image
  on the shared network. Provider keeps its existing alias router and authentication;
  the router now uses the original app's native load-job API and maps provider aliases
  to installed IDs. Native Qwen/Swift and Poly backends remain separate engine paths.
- Six installed configs: Ornith IQ3, Ornith IQ4, MiniCPM, uncensored MiniCPM, Gemma,
  Qwen/Swift. IQ3 boots by default. All imported weights/vision/MTP/pack paths checked.
- Both Ornith configs retain 262144 context capacity and q8_0 KV, 12 threads, CPU
  vision and existing general sampling defaults. Coding aliases set temperature .6
  and presence penalty 0; explicit request values win over router defaults.
- The two Bartowski Q4_K_M shards and dedicated mmproj were deleted, freeing exactly
  79,830,461,312 bytes. Other Q4_K_M models (MiniCPM/Gemma) and the AtomicChat IQ3/IQ4
  files were preserved. The retired config is unregistered and no longer mounted.
- Original UI: loopback10110 and the legacy model-screen address loopback10104.
  Manage models now loads a ready local config outside the download catalog without
  applying the unrelated catalog's settings. No download is triggered by this action.

## Verification

The real authenticated provider smoke requested HAZIR with thinking off and up to
32 tokens, checked nonempty responses AND the actual ready native model after every
call. All six physical models and the four Ornith aliases returned HTTP200 and HAZIR;
the final call returned to IQ3. Full timings are in provider-smoke.json; they include
switch/load time and are not decode TPS benchmarks. Vision/tool execution, long-context
quality and long-running concurrency were not benchmarked during this migration.

The first attempted native switch exposed a real integration mismatch: a forced
private port8080 was checked before the old instance was unloaded. The router now
omits that field and leaves free-port selection to the original app. Its gateway
resolves the selected instance's actual port. The native fake contract was updated
to reject fixed-port requests so this regression is covered.

Router: 9 tests passed, including native schema, alias remap, external unload health,
auth, unknown-model rejection, failed job propagation, streaming/draining and the
existing PolyStrata/default-sampling regressions. Controller registry: 17 tests passed.
App TypeScript typecheck/build passed; 20 existing native UI checks passed. The
historical deploy_smoke.sh was not run: it targets removed orchestration services
and would start unrelated workloads. Real provider switching is the smoke test here.

Manual browser verification on the existing loopback10104 model UI address:
Manage models -> Load IQ4 completed and showed IQ4 Ready/Loaded; Load IQ3 then
completed and restored IQ3 Ready/Loaded. The original app's model registry lists
all six configs and no Bartowski Ornith Q4_K_M entry. Both imported quant buttons
are enabled when idle. Container controller registry was also checked live: it
points to forgeloop-ai-strata and the native-aware router health, with actions[].
