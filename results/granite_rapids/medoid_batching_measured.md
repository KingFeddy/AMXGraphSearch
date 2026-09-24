# Medoid Hop-0 Batch Controller — Measured Results

End-to-end measurement of the implemented medoid-block BF16 GEMM batch
controller. This is the built-and-measured counterpart to the projections
in corrected_amdahl_analysis.md. Full discussion in [research log, Phase 9](../../docs/research-log.md#phase-9-medoid-hop-0-batch-controller--implementation-and-measurement).

## Design (one line)
Phase 1 (serial): one BF16 cblas_gemm_bf16bf16f32 over all N queries x R
medoid neighbors at hop 0. Phase 2 (parallel): each query seeds its
priority queue with the precomputed hop-0 distances and traverses from hop 1.

## arch_prctl confirmed firing
MKL_VERBOSE shows GEMM_BF16BF16F32 dispatching; steady-state hop-0 GEMM is
63-78us for all 1000 GIST1M queries. The first call carries a ~80ms one-time
AMX init, absorbed by an untimed warmup.

## Phase timing (GIST1M, T=32)
| Region                          | Time   | % of runtime |
|---------------------------------|--------|--------------|
| Phase 1 (setup + GEMM + scatter)| 0.43ms | 0.9%         |
| Phase 2 (traversal, hops 1+)    | 47.4ms | 99.1%        |

The AMX-addressable region is ~0.9% of runtime — the characterization's
central bound, confirmed in the working system.

## End-to-end results (30 interleaved iterations, mean +/- sample stddev)
| Dataset | T  | Baseline QPS    | Medoid-batch QPS | Speedup | Noise band | Verdict             |
|---------|----|-----------------|-----------------|---------|------------|---------------------|
| GIST1M  | 1  | 912 +/- 3       | 919 +/- 3       | 1.007x  | +/-0.50%   | faster beyond noise |
| GIST1M  | 32 | 20,697 +/- 436  | 20,638 +/- 377  | 0.997x  | +/-2.8%    | within noise        |
| SIFT1M  | 1  | 4,448 +/- 4     | 4,468 +/- 3     | 1.004x  | +/-0.13%   | faster beyond noise |
| SIFT1M  | 32 | 108,154 +/- 1129| 105,779 +/- 7949| 0.978x  | +/-7.6%    | within noise        |

## Recall preserved
| Dataset | Baseline | Medoid-batch | Drop  |
|---------|----------|---------|-------|
| GIST1M  | 88.31%   | 88.20%  | 0.11% |
| SIFT1M  | 99.11%   | 99.11%  | 0.00% |

BF16 applied only at hop 0, so the measured drop is far below the simulated
full-search BF16 estimate (0.90%, see amx_precision_findings.md).

Note on baseline figures: the 88.31% (GIST1M) and 99.11% (SIFT1M) baselines
here are from the medoid-batch comparison run at T=32, L=100. The threading-baseline
files (gist_baseline.md, vamana_baseline.md) report slightly different recall
(88.50%, 99.14%) because they are separate peak-throughput sweeps across thread
counts; the small differences are run/config variance, not an algorithmic
change. The comparison always measures baseline and the medoid-batch controller in the
same run, so the reported speedup and recall-drop are internally consistent.

## Conclusion
The controller is correct and the AMX kernel fires at the validated 3.52x,
but end-to-end speedup is small and statistically significant only at low
concurrency (T=1). At T=1 the measured 1.007x (GIST1M) matches the predicted
Amdahl ceiling (1.008x); the dominant low-thread mechanism is memory
amortization, not the compute acceleration the Amdahl fraction models. At
T=32 the effect is within noise because query-level parallelism already
amortizes the medoid loads via cross-thread cache sharing. This is the
AMX-resistance thesis demonstrated end-to-end.
