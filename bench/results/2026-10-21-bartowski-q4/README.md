# Bartowski Ornith Q4_K_M attempted UI trial

2026-10-21, host RTX 5060 Ti 16 GB (GPU 1), original Strata app 0.2.2 with the
imported Poly engine. Old PolyStrata and the development engine stayed stopped.
No LiteLLM, Pi, router or other application integration changed.

## Acquisition

`bartowski/Ornith-1.5-122B-GGUF` at revision
`a8bc08fa33fc3a3e79380bf661a84688c7ea0a50`, both Q4_K_M shards and its own f16 mmproj.
See `provenance.json` for byte sizes and verified SHA256 values. Models live at
`/mnt/llm-models/ornith-1.5-122b-bartowski`.
This is regular Ornith, whereas the existing AtomicChat IQ3 is Uncensored; any future
quality comparison therefore changes both quantization and checkpoint variant.

## IQ3 baseline

A fresh chat in the app at http://127.0.0.1:10110 asked exactly:

> bir sesli asistan yapsaydin mimariyi nasil kurardin

Manual mode, reasoning off. The app adds its system prompt, yielding 92 prompt tokens.
Engine defaults: temperature 1, top_p .95, top_k 20, min_p 0, presence_penalty 1.5,
repetition_penalty 1; the app may apply its own response sampling overrides.
The engine uses Q8 KV, 262144 context capacity, 12 threads, window 8, lookup 3,
tail skip 0, 1024 MiB VRAM reserve, CPU vision and IQ CPU reuse.
No download or hash job was active during this request.

Result: 402 output tokens, 36.5 decode tok/s, 971.5 ms prompt, 11007.7 ms decode,
157 / 236 draft tokens accepted, cache hit rate .557. The answer covered ASR, NLU,
actions, state, TTS, wake word, errors and privacy. One run is not a quality benchmark.
The raw sanitized metrics are in `iq3-ui-metrics.json`.

## Q4 load failure

The separate imported config was selected as the boot model, with the same context,
KV and resource settings. It failed before becoming ready:

```text
strata-poly: blk.0.ffn_gate_exps.weight and blk.0.ffn_up_exps.weight have different types, which this engine does not run
```

The GGUF headers confirm 13 mixed gate/up layers (see `mixed-gate-up.json`).
Blocks 0–11 use Q4_K gate / Q6_K up; block 47 uses Q6_K gate / Q8_0 up.
`src/poly/engine.cpp` rejects these in `Engine::load_moe`. The CPU expert path and
GPU slot layout also assume one shared gate/up type, so removing the guard alone
would be incorrect. No engine change or re-quantization was attempted.

There is **no Q4 TPS or generated response**, and no speed/quality verdict. The
container default was restored to IQ3; downloaded files and the Q4 config were kept.
256K is only the configured capacity, not a completed long-context test.
