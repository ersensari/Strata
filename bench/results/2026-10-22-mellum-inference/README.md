# Mellum Poly inference — 2026-10-22

Owner-approved GPU validation of the experimental Mellum backend. The native
Strata host service was not restarted. Mellum uses the separate
`build-mellum-staging/strata-poly`; the native Qwen and previous Poly binaries
were not replaced. The model is registered as `mellum-12b` and left selected in
the UI. The startup default and provider routing were not changed.

## Speed

RTX 5060 Ti 16 GB, physical GPU 1 only; 12 CPU threads; 128K configured context;
FP16 KV; all 1792 routed experts resident in VRAM (5.6 GiB of expert weights).
No MTP weights exist in this GGUF. Temperature 0, thinking off, same voice
architecture question used in the earlier Ornith local battery, warmed files,
one excluded speed warmup, three measured responses of 512 generated tokens.

| Configuration | Decode tok/s | Median |
| --- | --- | --- |
| Window 1, lookup 0, no draft | 194.8 / 194.8 / 194.5 | 194.8 |
| Window 8, lookup 3, no draft | 200.5 / 200.6 / 200.5 | 200.5 |

Lookup improved the median by 2.9%, but its text differed from the baseline.
It is previous-text speculation, not MTP: the API's `draft_n` counter includes
lookup guesses. The final selected config restores window 1 / lookup 0.

These are server decode timings, not end-to-end application performance or a
comparison on the user's voxel project. The 512-token speed responses were
truncated and contain reasoning-like text despite thinking-off prompting; they
are not scored as complete voice-architecture answers. Tokenization differs
between model families. No throughput claim is made for 128K input.

## Functional quality

Five automatic tasks, three trials each with thinking off:

| Task | Passed |
| --- | --- |
| Merge intervals, without changing input | 0/3 |
| LRU cache | 3/3 |
| Lexicographically smallest topological sort | 3/3 |
| Plain JSON and exact invoice total | 0/3 |
| Weather tool call | 3/3 |

The merged intervals were computed, but the generated code sorted the input in
place, violating the explicit requirement. Invoice JSON was well formed but the
total was 421.1 instead of 310.7. The voice-architecture answer contained
reasoning-like prose, weak Turkish and reached its 1536-token cap. This is a
small local battery, not a standard coding benchmark or a general model ranking.

A separate thinking-on followup tested only the two failed tasks, once each:
both passed. The invoice total was 310.7, and merge code preserved its input.
Thinking characters were separated into `reasoning_content`. This does not turn
the original 9/15 result into a 15/15 score.

## Backend validation

- Same GGUF, same token IDs and greedy decoding were compared with the existing
  official llama.cpp CUDA image, sequentially on GPU 1, FP16 KV and flash attention
  disabled on the reference. Tokenizer IDs matched on all four prompts.
- The four short continuations matched token-for-token: arithmetic (2 tokens),
  invoice prefix (16), code prefix (16), and a 1728-token prompt crossing the
  1024-token SWA boundary (6). This is not full numerical parity.
- First-token top-10 overlap was 9–10 of 10. The largest probability difference
  among common top-10 IDs was 0.1582 on the code prompt; the cause is unresolved.
  The backend remains experimental and must not be claimed bit-identical to
  llama.cpp or fully validated for long contexts.
- Two sequences, 16 greedy tokens each: batched results matched standalone runs.
- A/B/A cache restore: A returned the same text; the third request reused 1483
  prompt tokens instead of rereading the whole prompt.
- The temporary reference container was removed after verifying project ownership
  labels. Other GPU 0 services were not changed.

## Evidence

`quality-speed.json`, `thinking-check.json`, `lookup-speed.json`,
`reference-cases.json`, `parity-summary.json`, `batch-parity.log`,
`cache-restore.json` and `summary.json` retain the measured prompts, responses,
settings and timings. `first-load.json` and the Poly logs include raw token IDs
and the first generated probabilities. The runner is `bench/ornith_compare.py`
with `--expected-context 131072 --expected-kv fp16`.
