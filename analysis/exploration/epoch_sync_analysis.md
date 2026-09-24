# Epoch-Synchronous Search — Full Analysis and Retraction

(Explored as an alternative to medoid hop-0 batching. Note: this is distinct from
dynamic node-level batching, which is analyzed separately.)

## Initial Hypothesis
Restructure DiskANN's search loop so all queries advance one hop per epoch
in lockstep, enabling one BF16 GEMM per epoch covering all queries.

Initial (wrong) projection: 1.449x end-to-end on GIST1M vs the medoid hop-0 batching
projected 1.034x full-window ceiling. (The medoid-batch *measured* end-to-end
result is ~1.007x at T=1 — see [research log, Phase 9](../../docs/research-log.md#phase-9-medoid-hop-0-batch-controller--implementation-and-measurement). The 1.192x figure used in
earlier drafts was a superseded projection.)

## The Critical Error
The initial analysis assumed each epoch produces a 1000×32×960 matrix
(all 1000 queries expanding the same nodes simultaneously). This is wrong.

After hop 0, queries expand DIFFERENT nodes. The actual matrix at each epoch
is (queries sharing that specific node) × 32 × 960. From the hop log:

| Hop | Sharing% | Avg batch/node | AMX speedup |
|-----|----------|----------------|-------------|
| 0   | 99.9%    | 1000           | 3.52x       |
| 1   | 97.0%    | 33.8           | 2.00x       |
| 2   | 68.8%    | 3.2            | 1.02x       |
| 3+  | <37%     | ~1.1-1.6       | 1.02x       |

Weighted AMX across all hops: **1.063x** (not 3.51x as initially assumed).

## Corrected Amdahl (GIST1M)
- GIST1M search-only distance compute fraction: ~74%
- Weighted AMX 1.063x across all hops
- End-to-end: 1 / (0.26 + 0.74/1.063) = **1.026x**
- Barrier overhead at T=32: 1.5% (measured)
- Realistic: **1.011x**

## Conclusion
**RETRACTED.** Epoch-synchronous search does not beat medoid hop-0 batching.
Root cause: graph search diverges queries after hop 0. Epoch-sync adds
104 barrier synchronization points without meaningfully increasing the
per-node batch size beyond what already exists at hop 0.

The divergence after hop 0 is fundamental to how best-first search works
and cannot be overcome by scheduling policy alone.

## Barrier Overhead Measurement
From barrier_overhead_test.cpp (measured on Granite Rapids):
| Threads | Time/barrier(us) | 104 barriers(ms) | Search time(ms) | Overhead% |
|---------|-----------------|------------------|-----------------|-----------|
| 32      | 35.24           | 3.665            | 243.0           | 1.5%      |
| 64      | 68.14           | 7.087            | 47.0            | 15.1%     |
| 128     | 76.12           | 7.917            | 35.0            | 22.6%     |

Even though barrier overhead is only 1.5% at T=32, the weighted AMX
speedup collapses to 1.02x — not enough to justify the implementation.
