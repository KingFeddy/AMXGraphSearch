# AMX Precision Findings — Critical Discovery

## The arch_prctl Requirement (MOST IMPORTANT FINDING)

AMX tile registers require explicit OS-level permission before use. Without
the following syscall, MKL silently falls back to AVX-512 regardless of
data type or matrix size:

    syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)

This must be called ONCE at process startup before any AMX GEMM operations.

Verification:
- Without arch_prctl: VTune shows avx512_gemm_bf16bf16f32_generic in libmkl_avx512.so.3
- With arch_prctl: VTune shows _gemm_bf16bf16f32 in libmkl_rt.so.3 (AMX dispatcher)
- 4.92x speedup at 2048³ confirms AMX tiles firing (AVX-512 BF16 max is ~2x)

IMPLICATION: Any prior benchmark using cblas_gemm_bf16bf16f32 without
arch_prctl is measuring AVX-512 BF16, not AMX. This likely explains why
some published AMX benchmarks underreport the hardware's capability.

## FP32 Does Not Use AMX

cblas_sgemm (FP32) never dispatches to AMX regardless of matrix size.
AMX TMUL supports only BF16, INT8, and FP16 (Granite Rapids).
MKL dispatches FP32 GEMM to AVX-512 always.

Our original amx_gemm_test.cpp used cblas_sgemm — the "47x speedup"
was MKL AVX-512 vs naive loop, not AMX. Fixed in current version.

## Three-Way Speedup (with arch_prctl, T=1, GIST-like 960d, batch=1000)
| Path               | Time(ms) | vs FP32 GEMV |
|--------------------|----------|--------------|
| Naive triple loop  | 507.0    | 1.0x         |
| FP32 SGEMM (AVX512)| 0.503    | ~1007x       |
| BF16 GEMM (AMX)    | 0.143    | ~3545x       |
| AMX vs AVX-512     | —        | 3.52x        |

## AMX Dispatch Threshold (Matrix Size)
MKL only dispatches to AMX tiles above a minimum matrix size.
From amx_init_test.cpp sweep (arch_prctl enabled):
- 32×1000×128 (ANNS hop-0 SIFT): routes to AVX-512 BF16
- 32×1000×960 (ANNS hop-0 GIST): 3.52x → AMX firing
- 1024×1024×1024 (LLM-style): ~4.1x → AMX firing strongly
- 2048×2048×2048: ~5.0x → AMX firing at full throughput

The K dimension (DIM=960) is what pushes GIST1M into AMX territory.
SIFT1M at 128d stays in AVX-512 even with arch_prctl.

## BF16 Recall Impact on GIST1M
Simulated BF16 quantization (truncate lower 16 mantissa bits of FP32):
- Max absolute difference in query vectors: 0.003887
- Recall@10 FP32: 88.31%
- Recall@10 BF16-simulated: 87.41%
- Recall drop: 0.90% — ACCEPTABLE for production use

Traversal path is stable under BF16: avg dist comparisons nearly identical
(2545 vs 2547), confirming BF16 does not meaningfully change which nodes
get visited.

## Measured Recall in the Shipped Controller

The 0.90% above is the *simulated full-search* estimate (every distance
computed in BF16). The shipped medoid hop-0 batch controller applies BF16 only at
hop 0 (the medoid GEMM); all hop-1+ distances stay full-precision FP32.
The measured end-to-end recall drop is therefore much smaller:

| Dataset | FP32 baseline | Medoid-batch (BF16 hop-0) | Drop  |
|---------|---------------|----------------------|-------|
| GIST1M  | 88.31%        | 88.20%               | 0.11% |
| SIFT1M  | 99.11%        | 99.11%               | 0.00% |

Note: the controller uses BF16 for both datasets, not INT8 for SIFT. SIFT1M
is natively uint8, so an INT8 hop-0 path would be exactly lossless; that is
identified as future work. Empirically, BF16 on SIFT already shows 0.00%
measured drop. See [research log, Phase 9](../../docs/research-log.md#phase-9-medoid-hop-0-batch-controller--implementation-and-measurement) for the full measurement.
