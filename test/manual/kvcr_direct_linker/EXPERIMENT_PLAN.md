# KVCR linker experiment plan

This is a validation plan, not a performance result. CPU fake-transport tests do
not establish GPU correctness, overlap, UCX progress, or model latency.

## Fix the workload and environment first

Record SGLang/KVCR commits, model/tokenizer revisions, GPU and interconnect,
NIXL/UCX versions and settings, topology/ranks, page size, dtypes, exact prompts,
prompt lengths, output lengths, sampling seed, concurrency, and arrival schedule.
Use the same requests in the same order for paired comparisons; run warm-up
separately and repeat measured runs in alternating order. Keep model/kernel
settings, chunk sizes, preparation deadlines, and DRAM budget unchanged. Record
startup `restore_mode`, `nixl_backend`, `pool_capacities`, and the full config,
including `progressive_restore`. Keep compilation/kernel warm-up and peer
connection/metadata state matched within each comparison; cache-cold does not
have to mean an unwarmed model. Use unrelated prefixes for model warm-up.

Use deterministic greedy output where supported. Compare generated tokens and
logits against the same run with external restore disabled, using a declared
numerical tolerance where exact equality is inappropriate. First validate dense
attention, then representative DeepSeek hybrid and Mamba models. Include partial
prefixes, checkpoint boundaries, repeated requests, and batches of requests.

## Measure distinct residency states

| Case | Setup | Purpose |
| --- | --- | --- |
| Cold | Fresh target caches; no matching peer hint. | Full-prefill baseline and miss-path overhead. |
| HBM warm | Repeat the prefix while target device cache retains it. | Device-prefix reuse baseline; verify no restore occurred. |
| Local DRAM | Complete offload, evict only the target device prefix with controlled HBM pressure, and verify the KVCR objects remain resident. Use staged restore. | Host-to-device restoration without network transfer. |
| Peer staged | Source offload completed; target HBM and KVCR DRAM cold; matching peer hint; `direct_remote_restore=false`. | Peer-to-target-DRAM fetch followed by layer delivery. |
| Direct wait-all | Same source and cold target as peer staged; `direct_remote_restore=true`, `progressive_restore=false`. | Direct transfer cost with model execution held until all layers finish. |
| Direct progressive | Identical to direct wait-all except `progressive_restore=true`. | Isolate the benefit of releasing layers as their dependencies complete. |

Run both staged and direct cases with both values of `progressive_restore`.
Compare staged versus direct at the same progressive setting for the end-to-end
effect of bypassing target-DRAM staging. This also changes when peer work starts:
staged preparation fetches before admission, while direct transfer waits for
destination allocation. It does not isolate the cost of one staging copy.
Compare false versus true within one transfer mode to isolate progressive
release. Keep the restored prefix and chunk settings fixed and report transferred
bytes; hybrid staged fetch may retrieve more checkpoint/window bytes than direct
restoration. Recreate the target cache state between measured runs, while keeping
the source objects resident; repeating a peer request on a warmed target changes
the experiment. Confirm the intended source using KVCR transfer counters and
traces, not just the router hint or configured `restore_mode`.

## Instrumentation and interpretation

Enable `enable_telemetry` consistently across every paired instrumented run. Also
repeat a representative pair with telemetry disabled to quantify its overhead.
Record client TTFT, inter-token latency and throughput from individual request
traces; report distributions and repeat variability. Aggregate sum/count/max
metrics alone cannot provide percentiles.

Capture preparation queue, query, fetch, admission wait, restore wait/build/
submission/copy, offload timings, transferred bytes, operation high water,
backpressure, deadlines, misses, abandoned work, and uncertain loads. Compare
counter and histogram sum/count differences between snapshots from the same
process. Maxima are cumulative and cannot be differenced; use a fresh process
when comparing maxima. Do not subtract snapshots across reset or process changes.

`restore_layer_completion_seconds[layer]` measures elapsed wall time from the
start of restore submission until all transfers assigned to that logical layer
complete successfully while the batch remains successful. It is emitted once per
batch/layer, including wait-all runs; layers with no assigned copies have no
sample. It is not GPU compute-wait time and does not replace earlier-layer
barriers. This clock excludes descriptor construction but includes earlier
submission work and owner-thread completion-observation delay. Earlier-layer
samples remain if a later transfer fails, while samples after a batch failure
are suppressed; report failures and use failure-free windows for layer comparisons.
Inspect first, middle and final layer timings alongside final restore completion.
Confirm transfer/compute overlap with a short GPU profiler capture;
keep profiler runs separate from latency runs because profiling can perturb them.

GPU validation must confirm restored data and full-model outputs, early-layer
release, wait-all gating, source eviction behavior, and retained memory during
failed/incomplete transfers. Existing unit tests are useful preflight checks,
not substitutes for these checks.

## Attribute the commit checkpoints

The first six checkpoints are: local whole-object backend, peer staging,
DeepSeek C1/C2 support, Mamba checkpoint correctness, progressive named spans,
and sparse checkpoint capacity. The next three add direct restoration,
telemetry/diagnostics, and policy selection. Checkpoint 10 aligns successfully
offloaded ALL_PAGES sequences for tail-first LRU eviction; trailing windows and
single checkpoints are excluded.

Use external client timing and the same profiler tooling for comparisons before
telemetry exists. Detailed linker timings are available from checkpoint 8;
compare staged/direct and progressive/wait-all at that same instrumented head.
Do not compare an instrumented run with an uninstrumented run and attribute the
difference to the transfer change. Model-support commits are correctness gates,
not speedups over a predecessor that cannot run that model.

## Isolate sparse capacity

Hold the total DRAM budget, layout, workload, restore mode, eviction policy and
all timing settings fixed. Compare checkpoints 5 and 6 with the same KVCR
revision, staged restore, inherited LRU policy, compatible model and request
trace. Do not pass direct-restore, telemetry or policy-selection flags to these
earlier checkpoints, where those settings do not yet exist.
Record per-pool capacities, allocated bytes, resident objects, hit rate and
whether restored boundaries changed. Before sparse allocation, every pool has
`plan.page_capacity` objects; afterward inspect `plan.pool_capacities`. Use the
same external request measurements for this checkpoint comparison, since both
precede the telemetry commit. First use a working set that fits both
versions to separate correctness from capacity effects; then increase only the
working set to create pressure. Do not attribute a larger usable working set to
faster transfer code.

## Compare LRU and FIFO separately

After correctness and transfer comparisons, replay a fixed hot/cold request
trace whose working set exceeds DRAM capacity. Start each policy with the same
empty cache and use the same budget, warm-up, topology and restore mode. Change
only `eviction_policy` between `lru` and `fifo`; report misses, evictions,
transferred bytes, TTFT and throughput. Include a non-pressure control where both
policies should have the same hits. Keep these results separate from the
progressive-delivery and sparse-capacity comparisons.

Compare checkpoints 9 and 10 with LRU and the same pressure trace to isolate
sequence alignment. Measure retained prefix lengths as well as hit rate. Alignment
covers each offloaded segment; later accesses can change its recency again.

Publish the exact setup, per-run results and failed cases. A result applies only
to the tested model, workload and hardware until further measurements establish
otherwise.
