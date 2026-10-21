# Ornith IQ3 local comparison — 2026-10-22

Decision: keep Atomic AD-IQ3_XXS as the default. The mradermacher i1-IQ3_M
candidate was faster, but did not demonstrate better quality in this small test.
Both models remain installed and selectable. No provider aliases were changed.

| Measurement | Atomic Uncensored AD-IQ3_XXS | mradermacher original i1-IQ3_M |
| --- | --- | --- |
| Three warmed decode runs (tok/s) | 38.4, 38.5, 39.0 | 48.5, 48.8, 48.0 |
| Median decode (tok/s) | 38.5 | 48.5 |
| Automatic checks passed, three trials | 12/15 | 10/15 |
| Merge intervals | 3/3 | 3/3 |
| LRU cache | 3/3 | 3/3 |
| Lexicographically smallest topological sort | 3/3 | 1/3 |
| Plain JSON, exact schema and invoice total | 0/3 | 0/3 |
| Weather tool call | 3/3 | 3/3 |

The candidate's median was 26.0% higher. These are model/server decode timings,
not total request throughput or speech latency. Generated continuations differ.
The quality counts are a small local functional battery, not a general benchmark
score or proof of an intrinsic quantization or uncensoring effect.

## Conditions

- Same local Poly engine binary, physical NVIDIA GPU 1 (RTX 5060 Ti 16 GB),
  12 CPU threads, 262144 configured context, q8_0 KV, same expert placement
  profile at the start, same sampling and MTP settings.
- Temperature 0, top_p 0.95, top_k 20, min_p 0, presence penalty 0,
  repetition penalty 1, reasoning_effort none, actual process nice 0.
- Read all weight files before the final speed runs, exclude a 512-token
  speed warmup, then measure three 512-token responses to the exact question:
  `Eger bir sesli asistan yapsaydin nasil bir mimari kurardin?`
- Atomic measured no storage reads during those runs. Candidate read about
  4–16 MB per run, despite its preceding full-file cache warmup. This was not
  a cold-disk comparison. The runs were sequential on the same host.
- Configuring 256K does not test output quality at 256K: these prompts are short.
  Vision, long-context quality and sustained concurrent load were not tested.

## Quality observations

The Python cases execute in a labelled, networkless, read-only Docker sandbox.
They check mutation, duplicate edges, cycles, missing graph keys and cache edge
cases. The LRU check validates behavior; it does not empirically prove O(1).
Tool calls must have the expected name and arguments. JSON must parse without
Markdown, match the schema and total 310.70 TRY.

Atomic failed all invoice tasks with formatting, malformed JSON and/or wrong
totals. The candidate also failed all three. The candidate's topological sort
failed the lexicographic requirement once and duplicate-edge handling once.
Greedy outputs varied across repeated requests, so one successful answer should
not be treated as a guarantee.

Manual review of the full voice architecture answers found both usable as
general outlines, but neither covered acoustic echo cancellation or barge-in.
The candidate was longer, included endpointing and streaming, but also added
Kafka/Kubernetes without establishing a need, mistranslated far-field, and
proposed an unsubstantiated sub-500ms target. This did not establish a clear
quality improvement over Atomic.

## Evidence

- `atomic-iq3-speed-warm.json`: final warmed speed measurements.
- `atomic-iq3-final.json` and `atomic-iq3-quality-repeat.json`: the three Atomic
  quality trials. Their earlier, non-warmed speed measurements are not used in
  the final speed table.
- `mrad-i1-iq3m.json`: candidate, three quality trials and final warmed speeds.
- `atomic-iq3.json`: preliminary pilot at nice +5 with a concurrent download;
  excluded from the final comparison. Its topological-sort failure is retained.
- `provenance.json`: candidate revision, size and verified SHA-256.
- `tensor-layouts.json`: inspected quantization layouts.
- `../../ornith_compare.py`: exact prompts, validators and runner.

The original-base candidate and the Uncensored Atomic model differ in both base
weights and quantization layout. These results cannot isolate which caused the
speed or quality differences. Existing vision projector was reused for setup;
vision compatibility was not evaluated here.
