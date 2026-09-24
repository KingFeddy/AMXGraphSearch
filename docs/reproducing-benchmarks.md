# Reproducing the CPU benchmarks

Run the commands from the repository root. These are the Linux/Intel reference
commands used by the project; adjust the oneAPI library paths to your installation.
They are not a setup path for Wulver's NVIDIA GPUs.

## Requirements

- Linux on an Intel CPU with AMX BF16 support. Recorded CPU measurements used an
  Intel Xeon 6767P (Granite Rapids).
- A C++17 compiler with OpenMP, Intel oneAPI MKL, and the DiskANN build dependencies.
- DiskANN 0.7.0 with this repository's instrumentation and controller API patches.
  Faiss instrumentation targets v1.7.4 and is separate from the Vamana controller.
- A built, compatible DiskANN L2 float index plus binary queries and matching
  top-k ground truth. The example below expects these under `~/data/gist/`.

Datasets and built indexes are not included. The original study used SIFT1M and
GIST1M from [ANN Benchmarks](https://ann-benchmarks.com/) and a 100M-vector SIFT
subset. Follow the upstream data-format and index-building instructions for the
pinned DiskANN version. Dataset conversion scripts are not included in this
checkout. Index construction used `R=32`, `L_build=125`, `alpha=1.2` and 64 threads;
search comparisons used `L=100` and `K=10`.

## Prepare DiskANN

Apply both patches to a clean DiskANN 0.7.0 checkout before building the library:

- [`diskann_instrumentation.patch`](../vamana_instrumented/diskann_instrumentation.patch)
  adds traversal logging and its instrumentation header.
- [`diskann_medoid_hop0_api.patch`](../vamana_instrumented/diskann_medoid_hop0_api.patch)
  declares the controller entry point in `include/index.h`.

Use `git apply --check` before applying each patch, then build DiskANN according to
its own instructions. The commands below assume its headers are in `~/DiskANN/include`
and the built library is in `~/DiskANN/build/src`.

## Build and run the controller

```bash
source /opt/intel/oneapi/setvars.sh

g++ -O3 -march=native -fopenmp -std=c++17 \
    amx/medoid_batch_benchmark.cpp amx/medoid_batch_controller.cpp \
    -I$HOME/DiskANN/include -I/opt/intel/oneapi/mkl/latest/include \
    -L$HOME/DiskANN/build/src -L/opt/intel/oneapi/mkl/latest/lib \
    -L/opt/intel/oneapi/compiler/2026.0/lib \
    -ldiskann -lmkl_rt -liomp5 -lboost_program_options \
    -lpthread -lm -ldl -laio \
    -Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib \
    -Wl,-rpath,/opt/intel/oneapi/compiler/2026.0/lib \
    -o amx/medoid_batch_benchmark

# Run (GIST1M; T=1 is where the effect is measurable):
./amx/medoid_batch_benchmark \
    --index_path_prefix ~/data/gist/gist_index \
    --query_file ~/data/gist/gist_query.bin \
    --gt_file ~/data/gist/gist_groundtruth.bin \
    --K 10 --L 100 --T 1
```

Set `export MEDOID_ITERS=30` before running the benchmark to match the reported iteration
count. Run with `--T 1` and `--T 32` for the configurations in the overview.
`MEDOID_TIMING=1` adds controller phase timings for investigation; leave it unset
for the standard comparison.

The driver loads one index and query set for both paths, performs untimed warmups,
then interleaves baseline and batched runs. It prints QPS means and standard
deviations, Recall@10, a speedup ratio and a variability-based verdict. This
verdict uses a heuristic noise band, not a formal statistical significance test.

See the [recorded results](../results/granite_rapids/medoid_batching_measured.md)
for reference values. Hardware and runtime differences can change the measurements.

## Verify AMX execution

The benchmark driver requests Linux tile-state permission with
`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` before running GEMM. A failed
request stops this driver. Preserve this initialization when adapting the code.

Permission alone does not establish which kernel MKL selects. Inspect dispatch
and profiling evidence for the tested matrix shape. In the recorded environment,
GIST-like 960-dimensional shapes used AMX while the smaller 128-dimensional SIFT
shape used AVX-512 BF16.

[`amx_init_test.cpp`](../amx/amx_init_test.cpp) compares FP32 SGEMM and BF16 GEMM
across matrix shapes. It is a standalone kernel benchmark, separate from the
controller's complete-search measurements. See the
[precision and dispatch findings](../results/granite_rapids/amx_precision_findings.md)
for the recorded comparison.

## Instrumentation and analysis

The [Faiss patch and driver](../faiss_instrumented/) and
[DiskANN patch and driver](../vamana_instrumented/) record which neighbor vectors
each query evaluates at each traversal depth. `analysis/analyze_decay.py` expects
raw logs with `query_id`, `hop` and `neighbor_id` columns. The summary CSVs in
`results/` are already aggregated and are not raw inputs for that script.

To inspect the existing SIFT1M summaries without AMX hardware, follow the plotting
example in the [README](../README.md#explore-the-results). The plots' 5% sharing
threshold is an exploratory heuristic, not a hardware dispatch threshold.
