# KVCR direct linker

The KVCR linker transfers registered GPU KV pages directly to and from KVCR's
local DRAM or a hinted remote KVCR peer. It uses `deposit`, `query`, and `deliver`,
with at most four layer deliveries outstanding. Its owner thread continuously
polls KVCR, including while idle, so peer requests and transfer lifecycle events
keep progressing. Framework source-pin requests are declined; peers can retrieve
completed deposits in KVCR-owned DRAM. G3, `fetch`, and framework-owned source
serving are outside this integration.

Install KVCR with its NIXL/UCX dependencies in the serving environment. This
integration uses the public API inspected at KVCR commit `8a91e20`, including
the timeout ownership fix and PR #75 (labelled-key metadata caching).
[PR #64](https://github.com/ai-dynamo/kvcr/pull/64), Python allocation optimizations,
was still open at `0e33b521f35dbbb1cd28bf315f66f26382ab7252` and was absent from
that baseline. No performance result here assumes it is installed.

KVCR's local-copy timeout lifecycle fix is required: pending
native transfers must retain their handles and memory, report `uncertain` before
returning failure, and report `quiesced` only after native access stops. The
fix is present in `8a91e20`; GPU validation remains pending. Adapter-side
retention cannot repair a native transfer handle that KVCR has already
forgotten. KVCR is an external prerequisite and is not patched here.

An example worker configuration is:

```bash
python -m sglang.launch_server \
  --model-path zai-org/GLM-5.2-FP8 \
  --trust-remote-code --tp-size 8 --page-size 64 \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend kvcr \
  --hicache-storage-backend-extra-config '{
    "dram_size_gb": 8,
    "control_host": "0.0.0.0",
    "advertise_host": "192.0.2.10",
    "control_port": 19500,
    "nixl_port": 20500,
    "operation_timeout_ms": 1000,
    "abandon_timeout_ms": 5000
  }'
```

Replace `192.0.2.10` with an address reachable by the other workers. These options
are strict; unknown names fail startup.

| Option | Default | Meaning |
| --- | --- | --- |
| `kvcr_service_socket_path` | `/run/kvcr/memory.sock` | Guard service socket; `null` explicitly selects in-process DRAM. |
| `guard_index` | `0` | Base Guard index; the global rank is added. |
| `compatibility_digest` | Model/layout/shard namespace | Must match the Guard service's deployment compatibility digest. |
| `dram_size_gb` | `1` | Fallback in-process DRAM GiB per rank, rounded down to complete pages across all pools. |
| `control_host` | `0.0.0.0` | Address on which KVCR binds peer control. |
| `advertise_host` | Resolved local hostname | Address peers use to reach this worker. |
| `control_port` | `19500` | Base control port; the global rank is added. |
| `nixl_port` | `20500` | Base NIXL port; the global rank is added. |
| `operation_timeout_ms` | `1000` | Positive operation deadline in milliseconds. |
| `abandon_timeout_ms` | `5000` | Failure-reporting deadline, at least twice the operation timeout. |

Workers first claim Guard-backed memory through KVCR's public API. A missing
socket or refused connection falls back to anonymous DRAM inside the serving
process. An incompatible/busy Guard, permission error, or interrupted claim
fails startup: these do not prove that falling back is safe. KVCR owns Guard
claiming, recovery, and release; SGLang does not allocate a second DRAM pool
when a claim succeeds.

Provision the service using KVCR's `python -m kvcr.kvcr_service` command with
`--socket-path`, `--pool-dir`, `--guard-count`, `--pool-sizes-gb`, and
`--compatibility-digest`. Startup logs include the requested digest and ordered
pool layouts (name and block bytes). The service's ordered pool capacities, not
`dram_size_gb`, size Guard memory. Allocate a distinct Guard index per worker;
reserve separate index ranges and ports for colocated replicas. A service has
one digest across its Guards. For nonreplicated TP shards, set the same explicit
deployment digest on the service and all workers; change it when the model,
dtype, layout, page size, or parallel topology changes. Storage keys still retain
their individual shard namespaces.

In fallback mode, each pool receives capacity proportional to its bytes per
physical page, giving all pools the same page capacity. DRAM and GPU
registrations remain owned until successful KVCR shutdown. A shutdown error retains the mappings;
an uncertain transfer keeps its affected GPU pages unavailable for reuse until
quiescence is established.

As with other external Linkers, `flush_cache` drains transfers but does not
clear Guard storage. Recovered KV is valid only for unchanged model weights and
layout. Online weight replacement requires a fresh model/cache identity or
restarting with cleared storage and matching peers; flush alone is insufficient.

Remote hints use KVCR's versioned `kv.fetch` envelope, with a reachable
`source_control_endpoint` and uint64 `block_hashes`. Hints are advisory. The
lookup does not reserve source storage: capacity eviction between lookup and
delivery is an admitted load failure, including eviction caused by an older
offload. Failed admitted loads follow Linker's error path, without automatic
recomputation. Unsupported hint actions are ignored with a warning. The
runtime maps a storage hash's first 16 hexadecimal digits to the router hash.
With `cache_salt` or `extra_key`, existing KV events use a different hash chain;
those event hashes do not produce remote hits unless the router supplies the
corresponding storage hashes. Local reuse still works. Initial remote validation
covers replicated DSA/DeepSeek-V4 TP layouts, where every TP rank can use one
source endpoint. CP/PP or nonreplicated TP shards need hints targeting their
matching source shard; remote hints are disabled for those layouts. Local
reuse remains enabled. This adapter does not rewrite hint endpoints.

The initial layouts are DSA (`zai-org/GLM-5.2-FP8`) and DeepSeek-V4
(`sgl-project/DeepSeek-V4-Flash-FP8`), using the existing device-pool assembler.
Other model layouts, G3, and `--enable-linker-mla-dedup` are unsupported.

GPU validation is pending: local deposit/reload, peer direct delivery,
multiturn continuation, TP, idle source serving, concurrent load/offload,
Guard claim/recovery, timeout quarantine and shutdown, and layerwise overlap
must be checked with the fixed KVCR revision. CPU contract tests do not establish GPU correctness
or performance. `nvidia-smi` could not access the driver on this machine.

The SGLang base revision is `1814ece57f`; record the final SGLang and KVCR
revisions with every acceptance run, including dirty worktrees and whether
KVCR PR #64/#75 are present. Debug logs report per-layer readiness, total batch
drain time, and CPU `deliver()` submission time. Use a CUDA/PyTorch trace to
measure compute overlap; CPU readiness times alone cannot prove overlap.

Run the manual acceptance harness from the repository root:

```bash
python test/manual/cache/test_kvcr_linker.py --model glm52 --case local
python test/manual/cache/test_kvcr_linker.py --model dsv4 --case all --profile
```

GLM-5.2 uses TP8 and DeepSeek-V4 uses TP4. Peer tests need two disjoint groups
(16 or 8 GPUs respectively) and, when using Guard, at least that many Guards.
Start acceptance with fresh Guard pools, especially the destination's pools,
so an old local copy cannot masquerade as peer delivery. `--output-dir` retains logs, revision metadata,
and optional profiles. These commands have been checked for CLI/schema
compatibility but have not been run on GPU.

CPU validation on this branch: 62 shared-Linker tests passed (14 CUDA tests
skipped), 16 layout tests passed, 23 backend/runtime tests passed, and 24
registry tests passed. The Rust tree suite passed 923 tests (one existing
ignored test), and its Python bindings passed `cargo check`. Run each Python
file in its own process, as CI does:

```bash
python -m pytest test/registered/unit/mem_cache/test_unified_cache_linker.py -q
python -m pytest test/registered/unit/mem_cache/test_linker_pool_assembler.py -q
python -m pytest test/registered/unit/mem_cache/test_kvcr_direct_linker.py -q
python -m pytest test/registered/unit/mem_cache/test_registry.py -q
```
