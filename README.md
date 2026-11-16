# DSA5208 Project 3 - Parallel Training of a Mixture-of-Experts Network with MPI, OpenMP and CUDA

Mini-batch training of a mixture-of-experts (MoE) multilayer perceptron written from scratch in
C++17, with a serial version and a parallel version: the dense layers are data-parallel (gradient
all-reduce), the experts are expert-parallel (tokens exchanged with all-to-all), optimiser state is
sharded (ZeRO-1), with OpenMP threads (CPU) or CUDA (one GPU) inside each rank. All timings in
the report were measured on Vanda (CPU and A40 GPU nodes); Atlas ran the load-balance study and the tests.
No BLAS/cuBLAS: every matrix multiplication is our own code.

Everything described in the report is implemented and tested: serial and parallel training of
dense and MoE models, data parallelism (`MPI_Allreduce` or own ring all-reduce), expert parallelism
(`MPI_Alltoallv` or own pairwise `Isend/Irecv` exchange) with expert replicas when P > E, ZeRO-1,
OpenMP threads inside each rank, and a CUDA backend with our own kernels (one GPU per rank).

## Layout

| Path | Contents |
|---|---|
| `src/core/` | tensor, data loading (Fashion-MNIST, synthetic), timers, argument parsing |
| `src/backend/` | compute kernels: `cpu_omp.cpp` (blocked GEMM + OpenMP), `cuda/cuda_backend.cu` (own CUDA kernels, unified memory) |
| `src/parallel/` | communicators (`Comm_split`), all-reduce and all-to-all variants, ZeRO-1 |
| `src/serial/main_serial.cpp` | serial executable (no MPI, one thread) |
| `src/mpi/main_parallel.cpp` | parallel executable |
| `bench/` | `pingpong.cpp` (latency/bandwidth), GEMM benchmark |
| `tests/` | `make test`: GEMM, finite-difference gradient check, all-reduce / reduce-scatter / all-gather and all-to-all variants (1-8 ranks), serial vs parallel equivalence incl. expert parallelism, ZeRO-1 and checkpoints (double precision) |
| `scripts/` | data download, PBS job generation, log parsing, plotting |
| `pbs/` | Atlas and Vanda job templates; generated jobs in `pbs/generated/` |
| `results/` | `raw/` job logs (not submitted), `tables/` CSVs, `figures/` |

## Build and run locally (macOS)

Requires Homebrew `open-mpi` and `libomp`.

```bash
make                    # build/local-float/{mlp_serial, mlp_parallel, pingpong}
make test               # correctness tests in double precision
make data               # download Fashion-MNIST into data/fashion_mnist/
build/local-float/mlp_serial --epochs 3                      # Fashion-MNIST, Adam
OMP_NUM_THREADS=2 mpirun -np 4 build/local-float/mlp_parallel --epochs 3 --allreduce ring
OMP_NUM_THREADS=1 mpirun -np 8 build/local-float/mlp_parallel --epochs 2 --experts 16 --topk 2   # expert parallel, ep = 8
build/local-float/mlp_serial --data synthetic --steps 20 --batch 1024   # timing run
python scripts/sweep.py E1 --local                           # development run of experiment E1
```

## Build and run on Atlas

```bash
ssh <user>@atlas8.nus.edu.sg
source /app1/ebenv
module load gompi/2023a                 # GCC 12.3 + Open MPI 4.1.5 (not foss: no BLAS needed)
make PLATFORM=atlas && make PLATFORM=atlas REAL=double tests parallel serial
qsub pbs/generated/TEST/TEST_n1.pbs     # scripts generated on the laptop: python scripts/sweep.py TEST
qsub pbs/generated/CKPT/CKPT_n1.pbs     # checkpoint first; E3/E4/E5/E8 with -W depend=afterok:<CKPT job>
```

Atlas limits for our account (2 Nov): queue `parallel` only, at most 96 cores per job and per
user, nodes of 96 cores. `sweep.py` therefore packs all runs of an experiment into one job and
skips runs above 96 cores (128 and 256 cores go to Vanda).

After jobs finish: `make tables figures` (on the laptop, after copying `results/raw/` back).

## Build and run on Vanda

The login node is a virtual machine, so the builds run inside jobs (`sweep.py BUILD` for the CPU builds; for
the CUDA backend `pbs/vanda_gpu_test.pbs` builds it and runs the GPU correctness checks).

```bash
# inside a job (BUILD job, or interactive: qsub -I -P <project> -l select=1:ncpus=16:mem=64gb)
module load gompi/2023a                 # + CUDA/12.1.1 for the GPU build
make PLATFORM=vanda && make PLATFORM=vanda REAL=double tests parallel serial
make PLATFORM=vanda BACKEND=cuda && make PLATFORM=vanda BACKEND=cuda REAL=double tests parallel serial
python scripts/sweep.py E3 --platform vanda --project <project> --charged   # writes pbs/generated/vanda/E3/*.pbs
```

MPI options that work on Vanda (Ethernet only): `--mca pml ob1 --mca btl self,vader,tcp
--map-by ppr:<ranks per node>:node --bind-to none -x OMP_PROC_BIND=false`. Binding with
`--bind-to core` hangs in `MPI_Init` inside the job cpusets, and without `ppr` the ranks fill
the first node. `sweep.py` writes these options. GPU jobs: at most 2 A40 per job (one node).

## Command-line options

| Option | Meaning | Default |
|---|---|---|
| `--data` | `fashion` or `synthetic` | `fashion` |
| `--batch` | global mini-batch size B | 256 |
| `--hidden` | hidden width H | 512 |
| `--ffn` | FFN / expert inner width F | 1024 |
| `--blocks` | number of FFN (later MoE) blocks | 2 |
| `--experts` | experts per MoE block E (1 = dense FFN blocks) | 1 |
| `--topk` | experts per token k (MoE) | 2 |
| `--capacity` | capacity factor (MoE; ≤ 0 = no limit) | 1.25 |
| `--aux` | load-balancing loss weight (MoE) | 0.1 |
| `--groups` | capacity token groups; serial only (parallel sets ranks / EP size) | 1 |
| `--epochs` / `--steps` | train for whole epochs, or exactly this many steps (timing) | 1 / 0 |
| `--warmup` | steps excluded from timing | 5 |
| `--optim`, `--lr` | `adam` (default, lr 1e-3) or `sgd` (lr 0.05) | `adam` |
| `--seed` | weights, epoch permutations and synthetic data | 42 |
| `--log-every N` | MoE: print a `LOAD` line (max/mean load, CV, dropped, aux) every N steps | 0 (off) |
| `--dump FILE` | write final parameters, all experts in serial order (equivalence test, checkpoints) | - |
| `--load FILE` | start from a `--dump` file (same model and precision; timing runs, plan D17) | - |
| `--allreduce` | `mpi` or `ring` (parallel only) | `mpi` |
| `--ep` | ranks per expert-parallel group; must divide P and E (1 = data parallelism only, experts replicated) | min(P, E) |
| `--alltoall` | token exchange: `mpi` (`MPI_Alltoallv`) or `pairwise` (`Isend/Irecv`) (parallel, ep > 1) | `mpi` |
| `--a2a-wait` | barrier before each token exchange, timed as `t_a2a_wait` (load-imbalance waiting), so `t_a2a` is the exchange alone | off |
| `--zero` | ZeRO-1: reduce-scatter gradients, update this rank's shard (Adam state 1/P), all-gather parameters; uses the `--allreduce` variant | off |

Every run prints one `RESULT key=value ...` line per measurement; `scripts/parse_logs.py`
collects them into `results/tables/<experiment>.csv`.
