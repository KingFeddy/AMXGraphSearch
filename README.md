# AMXGraphSearch

Exploring Intel AMX acceleration for graph-based approximate nearest neighbor
search (ANNS). Built during NJIT's Honors Summer Research Initiative in 2026.

This repository contains C++ search instrumentation, a multithreaded batch
controller and performance analysis. It investigates where matrix-based distance
computation helps and how memory latency and diverging query paths limit the
benefit to a complete search.

**[Explore the controller](amx/medoid_batch_controller.cpp)** ·
**[Read the results](results/granite_rapids/medoid_batching_measured.md)** ·
**[Reproduce the benchmarks](docs/reproducing-benchmarks.md)**

## Research question

Vector search finds similar items by comparing their numerical representations.
Graph-based indexes speed up this process by following connections between
promising candidates instead of checking every stored vector.

Intel Advanced Matrix Extensions (AMX) can accelerate matrix multiplication on
supported CPUs. The question is whether graph searches share enough candidate
vectors to combine their distance calculations into useful matrix operations.

The study compares Faiss HNSW and DiskANN Vamana on SIFT1M and GIST1M, with
additional Vamana measurements on a 100M-vector SIFT subset. An IVF baseline
provides a comparison with a cluster-based index. The implemented batch
controller extends DiskANN/Vamana.

## How the controller works

Vamana queries start from a common entry node. The controller uses this shared
starting point before allowing each query to follow its own graph path.

1. Load the entry node's neighbor vectors once for the query batch.
2. Compute the first-hop distances together using BF16 GEMM with FP32
   accumulation, then initialize each query's candidate queue.
3. Run the remaining graph traversals independently across CPU threads using
   full-precision distance calculations.

```mermaid
flowchart LR
    A[Query batch] --> B[Shared first-hop distances]
    B --> C[Per-query candidate queues]
    C --> D[Parallel graph traversal]
    D --> E[Nearest-neighbor results]
```

GEMM is matrix multiplication. Batching lets queries reuse the same candidate
vector data while computing their own distances. OpenMP distributes the later
searches across threads; oneMKL selects the matrix kernel for the hardware and
matrix shape.

## What we found

- **Faster distance kernels:** the GIST-like 960-dimensional benchmark measured a
  **3.52× speedup** for AMX BF16 GEMM over AVX-512 FP32 SGEMM. This measures the
  matrix kernel rather than the complete search.
- **Memory latency limits acceleration:** VTune and Linux perf profiling showed
  stalls from irregular memory access and cache misses alongside low DRAM
  bandwidth utilization. Faster arithmetic cannot remove those waits.
- **Queries diverge after the shared start:** later traversals visit different
  nodes, leaving smaller groups of queries that can usefully be batched. The
  implemented controller batches only the first hop.
- **Matrix shape matters:** in the measured environment, the GIST-like shape
  used AMX while the smaller 128-dimensional SIFT shape used AVX-512 BF16.

### Complete-search performance

Recorded on an Intel Xeon 6767P (Granite Rapids) with `K=10` and `L=100`.
Values are mean queries per second over 30 interleaved runs per configuration.

| Dataset | CPU threads | Baseline QPS | Batch controller QPS |
| --- | ---: | ---: | ---: |
| SIFT1M | 1 | 4,448 | 4,468 |
| SIFT1M | 32 | 108,154 | 105,779 |
| GIST1M | 1 | 912 | 919 |
| GIST1M | 32 | 20,697 | 20,638 |

The single-thread improvements were about 0.4% on SIFT1M and 0.7% on GIST1M.
The 32-thread differences fell within the benchmark's reported variability band.
The full results include standard deviations and the comparison method.

In the 32-thread accuracy comparison, Recall@10 remained **99.11%** on SIFT1M
and changed from **88.31% to 88.20%** on GIST1M, a drop of 0.11 percentage points.
Recall@10 measures how many of the true ten nearest neighbors are recovered.

The GIST1M phase measurements put first-hop setup and computation at about
0.9% of runtime at 32 threads. Accelerating that small region leaves most search
time unchanged. The findings establish both a working batching implementation
and the limits of its impact on overall performance.

See the [controller measurements](results/granite_rapids/medoid_batching_measured.md)
and [kernel measurements](results/granite_rapids/amx_precision_findings.md).

## Implementation highlights

- **Search instrumentation:** C++ patches record query IDs, traversal depths and
  evaluated neighbors in Faiss and DiskANN to measure sharing between queries.
- **Batch controller:** a two-phase DiskANN extension combines shared first-hop
  computation with independent traversal and per-thread search state.
- **Benchmark driver:** compares the baseline and controller on the same index
  and queries with warmup, interleaved runs and recall checks against ground truth.
- **Analysis:** Python scripts summarize sharing across traversal depths, plot
  results and examine numerical precision and workload effects.

## Tech stack

| Area | Technologies |
| --- | --- |
| Search implementation | C++17, OpenMP, DiskANN 0.7.0, Faiss v1.7.4 |
| Matrix computation | Intel AMX, oneMKL, BF16 GEMM, FP32 SGEMM |
| Analysis | Python, pandas, NumPy, Matplotlib |
| Profiling | Intel VTune, Linux perf |
| CPU measurements | Linux, Intel Xeon 6767P (Granite Rapids) |

The broader research also explored GPU counterparts on NJIT's Wulver HPC cluster
using NVIDIA GPUs and Slurm job scheduling. This checkout contains the CPU
implementation and recorded CPU results; GPU implementations, Slurm scripts and
CPU/GPU comparison measurements are not included here.

## Explore the results

The recorded results can be read without AMX hardware. To regenerate the
HNSW/Vamana sharing plot from the included SIFT1M summaries, run these commands
from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy matplotlib
python analysis/plot_combined.py \
  results/faiss_decay.csv \
  results/vamana_decay.csv \
  /tmp/amxgraphsearch-sharing.png
```

The plot shows how candidate-vector sharing changes with traversal depth. Its
5% reference line is a research heuristic, not an AMX hardware threshold.

Running the C++ benchmarks requires a compatible Linux/Intel system, oneAPI MKL,
a patched DiskANN build and prepared datasets. Follow the
[reproduction guide](docs/reproducing-benchmarks.md) for the reference build
command, input requirements and AMX initialization details.

## Repository guide

| Location | Contents |
| --- | --- |
| [`amx/`](amx/) | Batch controller, benchmark driver and matrix-kernel experiments |
| [`faiss_instrumented/`](faiss_instrumented/) | HNSW instrumentation patch and search driver |
| [`vamana_instrumented/`](vamana_instrumented/) | Vamana instrumentation, controller API patch and search driver |
| [`ivf_baseline/`](ivf_baseline/) | Cluster-based baseline for sharing analysis |
| [`analysis/`](analysis/) | Analysis scripts and exploration notes |
| [`results/`](results/) | CSV summaries, plots, profiling records and measured findings |
| [`docs/research-log.md`](docs/research-log.md) | Original research phases, projections and subsequent corrections |

## Research context

Developed during NJIT's Honors Summer Research Initiative in 2026 under the
guidance of Prof. Xiaoning Ding. Contributor:
[Frederick Rajakumar](https://github.com/KingFeddy).
The work builds on [Faiss](https://github.com/facebookresearch/faiss) and
[DiskANN](https://github.com/microsoft/DiskANN).

Further work includes evaluating additional datasets and a separate INT8
first-hop path. These are research directions rather than implemented features.
