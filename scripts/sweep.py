"""Generate PBS job scripts for one experiment from pbs/*_template.pbs.

    python scripts/sweep.py E3                -> pbs/generated/atlas/E3/*.pbs, prints qsub lines
    python scripts/sweep.py E3 --platform vanda --project NAME
                                              -> pbs/generated/vanda/E3/*.pbs: jobs up to 160 cores in
                                                 the free queue, larger ones charged to project NAME
    python scripts/sweep.py E3 --dry-run      -> only list the runs
    python scripts/sweep.py E1 --local        -> run serially on this machine (development only;
                                                 logs in results/raw/local_<exp>/, never reported)

The runs of one experiment are packed into one job per node count: the job
holds whole nodes and runs them one after another, each writing
results/raw/<exp>/<name>.log (copy back to results/raw/<platform>/<exp>/). Atlas lets one user run at most 96 cores at a
time and its queue is long, so one job per run would mostly wait. Runs larger
than the platform's per-user core limit are skipped (they go to Vanda).
parse_logs.py turns the logs into results/tables/<exp>.csv.
"""
import argparse
import os
import pathlib
import string
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Values checked on Atlas 2 Nov (plan "Atlas facts"): our account can use the
# `parallel` queue only, at most 96 cores per job and per user; nodes have 96 cores.
# Vanda (checked 2 Nov): CPU nodes of 72 cores; free queue auto_free -> cpu_parallel
# (<= 160 cores per user, charge 0); charged queue via "#PBS -P project" -> batch_cpu
# (<= 720 cores, 10 nodes per user). Same gompi/2023a toolchain.
PLATFORMS = {
    "atlas": {
        "queue_line": "#PBS -q parallel",
        "modules": "gompi/2023a",  # GCC 12.3 + Open MPI 4.1.5, no BLAS
        "cores_per_node": 96,
        "max_cores": 96,
        "free_max_cores": 96,
        "min_ncpus": 12,           # queue minimum per job
        "mem_per_node": "",
        "build": "build/atlas-float",
        "template": "atlas_template.pbs",
        "mpi_extra": "",
        # each rank gets its own T cores; OpenMP threads stay inside them
        "bind": "--map-by slot:PE={t} --bind-to core -x OMP_PROC_BIND=close -x OMP_PLACES=cores",
    },
    "vanda": {
        "queue_line": "#PBS -q auto_free",
        "modules": "gompi/2023a",
        "cores_per_node": 72,
        "max_cores": 720,
        "free_max_cores": 160,
        "min_ncpus": 2,
        "mem_per_node": ":mem=480gb",  # Vanda requires a memory request (sample job: 480gb for 72 cores)
        "build": "build/vanda-float",
        "template": "vanda_template.pbs",
        # UCX cannot be selected (RDMA ports down, Ethernet only) and the default
        # transport hung in MPI_Init on a single node (2 Nov); ob1 + TCP works and
        # has half the default's inter-node latency (16 vs 32 us)
        "mpi_extra": "--mca pml ob1 --mca btl self,vader,tcp",
        # --map-by slot:PE=T --bind-to core hangs in MPI_Init inside Vanda's job cpusets
        # (2-3 Nov, even for 1 rank); without it everything runs, so no binding here -
        # and no OMP_PROC_BIND either, or every rank would pin its threads to the same cores.
        # Ranks are spread evenly with ppr (per node); filling nodes by slot put e.g. all
        # 32 ranks x 8 threads of a 4-node job on the first node (3 Nov)
        "bind": "--map-by ppr:{ppr}:node --bind-to none -x OMP_PROC_BIND=false",
        # GPU jobs (E9, E10): free queue `gpu` allows 2 A40 (one node) and 36 cores per job
        "cuda_build": "build/vanda-cuda-float",
        "gpu_modules": "gompi/2023a CUDA/12.1.1",
        "gpu_select": "1:ncpus=36:mpiprocs=36:ngpus=2:mem=240gb",
    },
}
P = dict(PLATFORMS["atlas"])

POWERS_OF_TWO_TO_256 = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def mpi(ranks, threads=1):
    """mpirun with `threads` cores per rank (Open MPI): ranks are bound to their
    cores and OpenMP threads stay inside them; -x forwards the settings to
    every node."""
    nodes = -(-ranks * threads // P["cores_per_node"])
    return (f"mpirun -np {ranks} {P['mpi_extra']} {P['bind'].format(t=threads, ppr=-(-ranks // nodes))} "
            f"-x OMP_NUM_THREADS={threads}")


def run(name, ranks, threads, command, nodes=None):
    return {"name": name, "ranks": ranks, "threads": threads, "command": command, "nodes": nodes}


def runs_build():
    """Compile both precisions inside a job (the Vanda login node is a VM, L8 p47)."""
    plat = P["build"].split("/")[1].split("-")[0]
    yield run("build", 16, 1, f"make -j16 PLATFORM={plat} && make -j16 PLATFORM={plat} REAL=double tests parallel serial")


def runs_test():
    """Correctness on the cluster (plan §6): the double-precision test suite."""
    b = P["build"].replace("float", "double")
    yield run("test_gemm", 4, 1, f"{b}/test_gemm_serial && OMP_NUM_THREADS=4 {b}/test_gemm_omp")
    yield run("test_gradcheck", 1, 1, f"{b}/test_gradcheck")
    for p in [1, 2, 3, 4, 5, 8, 16]:
        yield run(f"test_collectives_p{p}", p, 1, f"{mpi(p)} {b}/test_allreduce && {mpi(p)} {b}/test_alltoall")
    yield run("test_equivalence", 16, 1, f'tests/test_equivalence.sh {b} "mpirun --oversubscribe {P["mpi_extra"]} -x OMP_NUM_THREADS=1"')


def runs_e0():
    """Ping-pong latency / bandwidth (alpha, beta for the cost model): two ranks
    on one node, and one rank on each of two nodes."""
    exe = f"{P['build']}/pingpong"
    yield run("pingpong_intra", 2, 1, f"{mpi(2)} {exe}")
    yield run("pingpong_inter", 2, 1, f"mpirun -np 2 {P['mpi_extra']} --map-by ppr:1:node {exe}", nodes=2)


def runs_e1():
    """Serial runtime vs problem size (grading item 2, plan E1), synthetic data,
    one core, MoE model (default E = 16, k = 2, H = 512, F = 2H, 2 blocks,
    capacity 2.0 so drops do not change the work). B, H, E and k are varied
    separately: compute grows with B, H^2 and k, while E adds parameters but
    only router work - the MoE analogue of L12 p37 (dimension vs nonzeros).
    The dense model's batch sweep and the GEMM benchmark are kept for reference."""
    exe = f"{P['build']}/mlp_serial"
    base = "--data synthetic --capacity 2.0 --warmup 2 --exp E1 --quiet"
    for b in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        yield run(f"e1_batch{b}", 1, 1, f"{exe} {base} --experts 16 --steps {12 if b <= 1024 else 6} --batch {b}")
    for h in [64, 128, 256, 512, 1024, 2048]:
        yield run(f"e1_hidden{h}", 1, 1, f"{exe} {base} --experts 16 --steps 8 --batch 256 --hidden {h} --ffn {2 * h}")
    for e in [2, 4, 8, 16, 32, 64, 128]:
        yield run(f"e1_experts{e}", 1, 1, f"{exe} {base} --experts {e} --topk 2 --steps 8 --batch 256")
    for k in [1, 2, 4, 8]:
        yield run(f"e1_topk{k}", 1, 1, f"{exe} {base} --experts 16 --topk {k} --steps 8 --batch 256")
    for blocks in [1, 2, 4, 8]:
        yield run(f"e1_blocks{blocks}", 1, 1, f"{exe} {base} --experts 16 --steps 8 --batch 256 --blocks {blocks}")
    for b in [256, 1024, 4096]:
        yield run(f"e1_dense_batch{b}", 1, 1, f"{exe} {base} --experts 1 --steps 8 --batch {b}")
    yield run("e1_gemm", 1, 1, f"OMP_NUM_THREADS=1 {P['build']}/gemm_bench --max-n 2048")


# Partly trained MoE model that the timing runs start from (plan D17): random
# and trained routers load the experts differently. 235 steps of B = 256 = one
# pass over 60 000 synthetic samples. Run before E3, E4, E5, E8.
CKPT_E16 = "results/ckpt/moe_E16_H512_F1024_L2.bin"


def runs_ckpt():
    yield run("ckpt_moe_E16", 16, 1,
              f"mkdir -p results/ckpt && {mpi(16)} {P['build']}/mlp_parallel --data synthetic --experts 16 "
              f"--batch 256 --steps 235 --exp CKPT --dump {CKPT_E16}")


def runs_e3():
    """Strong scaling (plan E3): fixed model (E = 16) and global batch 16384;
    capacity 2.0 so the limit does not change with P (D16); MoE runs start
    from the checkpoint (D17). Dense data-parallel model as the reference for
    the all-reduce share. 32 ranks x 3 threads gives the 96-core point."""
    exe = f"{P['build']}/mlp_parallel"
    moe = f"--data synthetic --experts 16 --capacity 2.0 --batch 16384 --load {CKPT_E16} --steps 15"
    # T_s for speedup: the serial build on the same problem (one capacity group)
    # same step window as the parallel runs (15 steps, 5 warm-up); an earlier
    # 8-step run (e3_serial) averaged the load over other steps
    yield run("e3_serial15", 1, 1, f"{P['build']}/mlp_serial {moe} --warmup 5 --exp E3")
    for p in POWERS_OF_TWO_TO_256:
        for variant in ["mpi", "ring"]:
            yield run(f"e3_p{p}_{variant}", p, 1,
                      f"{mpi(p)} {exe} {moe} --allreduce {variant} --alltoall mpi --a2a-wait --exp E3")
        yield run(f"e3_p{p}_dense", p, 1,
                  f"{mpi(p)} {exe} --data synthetic --experts 1 --batch 16384 --allreduce mpi --steps 15 --exp E3")
    yield run("e3_p32_t3_mpi", 32, 3, f"{mpi(32, 3)} {exe} {moe} --allreduce mpi --alltoall mpi --a2a-wait --exp E3")


def runs_e7():
    """ZeRO-1 on the CPU (plan E7), at P > E where expert replicas exist (D = P/16):
    the E3 problem with and without --zero, both all-reduce variants. Optimiser
    state per rank (optim_state_mb), optimiser time and communication time."""
    exe = f"{P['build']}/mlp_parallel"
    moe = f"--data synthetic --experts 16 --capacity 2.0 --batch 16384 --load {CKPT_E16} --steps 15 --a2a-wait --exp E7"
    for p in [64, 128, 256]:
        for variant in ["mpi", "ring"]:
            for zero in ["", " --zero"]:
                tag = "_zero" if zero else ""
                yield run(f"e7_p{p}_{variant}{tag}", p, 1, f"{mpi(p)} {exe} {moe} --allreduce {variant}{zero}")


def runs_e4():
    """Weak scaling (plan E4): 512 tokens per rank. (a) DP x EP grid: E = 16
    fixed, EP groups of min(P, 16) ranks, expert replicas grow with P; starts
    from the checkpoint. (b) E = P: one expert per rank, parameters grow with P,
    the all-to-all spans all ranks; random router (no checkpoint per E; with
    centred synthetic data the random router is close to balanced, plan §3.2).
    Dense DP is the reference for the all-reduce share."""
    exe = f"{P['build']}/mlp_parallel"
    common = "--data synthetic --capacity 2.0 --a2a-wait --steps 15 --exp E4"
    for p in POWERS_OF_TWO_TO_256:
        b = 512 * p
        yield run(f"e4_grid_p{p}", p, 1, f"{mpi(p)} {exe} {common} --batch {b} --experts 16 --load {CKPT_E16}")
        if p >= 2:
            yield run(f"e4_eq_p{p}", p, 1, f"{mpi(p)} {exe} {common} --batch {b} --experts {p} --ep {p}")
        yield run(f"e4_dense_p{p}", p, 1, f"{mpi(p)} {exe} {common} --batch {b} --experts 1")


def runs_e5():
    """Communication variants (plan E5): all-reduce MPI vs ring over message
    sizes 8 B - 64 MB, and the E3 problem with MPI_Alltoallv vs pairwise
    Isend/Irecv token exchange, 16-256 ranks (with --a2a-wait, so t_a2a
    compares the exchanges alone)."""
    for p in [16, 32, 64, 128, 256]:
        yield run(f"e5_allreduce_p{p}", p, 1, f"{mpi(p)} {P['build']}/allreduce_bench")
        for a2a in ["mpi", "pairwise"]:
            yield run(f"e5_a2a_{a2a}_p{p}", p, 1,
                      f"{mpi(p)} {P['build']}/mlp_parallel --data synthetic --experts 16 --capacity 2.0 "
                      f"--batch 16384 --load {CKPT_E16} --alltoall {a2a} --a2a-wait --steps 15 --exp E5")


def runs_e8():
    """Ranks x threads (plan E8, L9 p22), E3 problem: every power-of-two split
    of one node, and four splits of 256 cores."""
    exe = f"{P['build']}/mlp_parallel"
    common = f"--data synthetic --experts 16 --capacity 2.0 --batch 16384 --load {CKPT_E16} --a2a-wait --steps 15 --exp E8"
    c = P["cores_per_node"]
    r = 1
    while r <= c:
        yield run(f"e8_node_r{r}_t{c // r}", r, c // r, f"{mpi(r, c // r)} {exe} {common}")
        r *= 2
    for r, t in [(256, 1), (128, 2), (64, 4), (32, 8)]:
        yield run(f"e8_256_r{r}_t{t}", r, t, f"{mpi(r, t)} {exe} {common}")


def runs_e6():
    """Load balance (plan E6): aux weight x capacity factor x k on Fashion-MNIST,
    one EP group of 16 ranks (E = 16). Per run: final accuracy, max/mean and CV
    of the expert load, dropped fraction (RESULT), the same per epoch (EPOCH)
    and every 47 steps (LOAD, ~5 per epoch at B = 256). The main comparison
    (aux 0 vs 0.1, cf 1.25, k 2) is repeated with 3 seeds for the accuracy."""
    base = f"{mpi(16)} {P['build']}/mlp_parallel --data fashion --experts 16 --epochs 5 --log-every 47 --a2a-wait --exp E6"
    for aux in [0, 0.01, 0.1]:
        for cf in [1.0, 1.25, 2.0]:
            for k in [1, 2]:
                seeds = [42, 43, 44] if (aux in (0, 0.1) and cf == 1.25 and k == 2) else [42]
                for seed in seeds:
                    yield run(f"e6_aux{aux}_cf{cf}_k{k}_s{seed}", 16, 1,
                              f"{base} --aux {aux} --capacity {cf} --topk {k} --seed {seed}")


def runs_e11():
    """Data conclusions (plan E11): one full Adam run on Fashion-MNIST, 64 ranks
    (EP groups of 16, 4 expert replicas), 10 epochs. Test accuracy per epoch,
    confusion matrix and class -> expert routing after training."""
    yield run("e11_full", 64, 1,
              f"{mpi(64)} {P['build']}/mlp_parallel --data fashion --experts 16 --epochs 10 --log-every 47 --exp E11")


def runs_e9():
    """GPU vs CPU per step (plan E9): one A40 (CUDA backend) and one CPU core of
    the same node, same problem, vs batch B (H = 512, from the checkpoint) and
    vs width H (B = 4096, random init). The whole-CPU-node reference is E8."""
    gpu, cpu = f"{P['cuda_build']}/mlp_serial", f"{P['build']}/mlp_serial"
    for b in [1024, 4096, 16384, 65536]:
        common = f"--data synthetic --experts 16 --capacity 2.0 --batch {b} --load {CKPT_E16} --exp E9"
        yield run(f"e9_gpu_b{b}", 1, 1, f"{gpu} {common} --steps 12 --warmup 2")
        yield run(f"e9_cpu_b{b}", 1, 1, f"{cpu} {common} --steps {4 if b >= 16384 else 8} --warmup 1")
    for h in [128, 256, 1024, 2048]:
        common = f"--data synthetic --experts 16 --capacity 2.0 --batch 4096 --hidden {h} --ffn {2 * h} --exp E9"
        yield run(f"e9_gpu_h{h}", 1, 1, f"{gpu} {common} --steps 12 --warmup 2")
        yield run(f"e9_cpu_h{h}", 1, 1, f"{cpu} {common} --steps {4 if h >= 1024 else 8} --warmup 1")


def runs_e10():
    """MPI + CUDA (plan E10): 1 and 2 GPUs (one rank per GPU), data parallel
    (--ep 1, experts replicated) vs expert parallel (--ep 2), with the
    all-to-all waiting split off; ZeRO-1 on the data-parallel layout."""
    exe = f"{P['cuda_build']}/mlp_parallel"
    for b in [16384, 65536]:
        common = f"--data synthetic --experts 16 --capacity 2.0 --batch {b} --load {CKPT_E16} --a2a-wait --steps 15 --exp E10"
        yield run(f"e10_g1_b{b}", 1, 1, f"{mpi(1)} {exe} {common}")
        yield run(f"e10_g2_dp_b{b}", 2, 1, f"{mpi(2)} {exe} {common} --ep 1")
        yield run(f"e10_g2_ep_b{b}", 2, 1, f"{mpi(2)} {exe} {common} --ep 2")
        yield run(f"e10_g2_dp_zero_b{b}", 2, 1, f"{mpi(2)} {exe} {common} --ep 1 --zero")


# name -> (runs, exclusive nodes (timing), default walltime[, GPU job])
EXPERIMENTS = {
    "BUILD": (runs_build, False, "00:20:00"),
    "TEST": (runs_test, False, "00:30:00"),
    "E0": (runs_e0, False, "00:15:00"),
    "E1": (runs_e1, True, "01:00:00"),
    "CKPT": (runs_ckpt, False, "00:30:00"),
    "E3": (runs_e3, True, "02:00:00"),
    "E4": (runs_e4, True, "01:00:00"),
    "E5": (runs_e5, True, "01:00:00"),
    "E6": (runs_e6, False, "03:00:00"),
    "E8": (runs_e8, True, "01:00:00"),
    "E11": (runs_e11, False, "01:00:00"),
    "E7": (runs_e7, True, "00:30:00"),
    "E9": (runs_e9, False, "01:30:00", True),
    "E10": (runs_e10, False, "00:45:00", True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp", choices=sorted(EXPERIMENTS))
    parser.add_argument("--walltime", help="override the experiment's default")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local", action="store_true", help="run here instead of writing PBS scripts")
    parser.add_argument("--platform", choices=sorted(PLATFORMS), default="atlas")
    parser.add_argument("--project", help="Vanda project charged for jobs above the free-queue limit")
    parser.add_argument("--charged", action="store_true",
                        help="charge every job to --project (same node pool for all core counts)")
    args = parser.parse_args()
    P.update(PLATFORMS[args.platform])
    if args.local:
        P["build"] = "build/local-float"
        P["cores_per_node"] = P["max_cores"] = os.cpu_count()
    generate, exclusive, walltime, *gpu = EXPERIMENTS[args.exp]
    gpu = bool(gpu and gpu[0])
    if gpu and "gpu_select" not in P:
        raise SystemExit(f"{args.exp} needs GPUs: use --platform vanda")

    # group the runs that fit the per-user limit by node count
    groups = {}
    for r in generate():
        cores = r["ranks"] * r["threads"]
        limit = P["max_cores"] if args.project or args.local else P["free_max_cores"]
        if cores > limit:
            print(f"{r['name']}: {cores} cores > {limit} allowed here - skipped")
            continue
        nodes = 1 if gpu else r["nodes"] or -(-cores // P["cores_per_node"])
        groups.setdefault(nodes, []).append(r)
        print(f"{r['name']}: nodes={nodes} ranks={r['ranks']} threads={r['threads']} :: {r['command']}")
    if args.dry_run:
        return

    # on the cluster: results/raw/<exp>/; copied back to results/raw/<platform>/<exp>/
    rel_logs = pathlib.Path("results") / "raw" / (f"local_{args.exp}" if args.local else args.exp)
    (ROOT / rel_logs).mkdir(parents=True, exist_ok=True)
    if args.local:
        for runs in groups.values():
            for r in runs:
                with open(ROOT / rel_logs / f"{r['name']}.log", "w") as log:
                    subprocess.run(r["command"], shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return

    template = string.Template((ROOT / "pbs" / P["template"]).read_text())
    out_dir = ROOT / "pbs" / "generated" / args.platform / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    for nodes, runs in sorted(groups.items()):
        if exclusive:
            per_node = P["cores_per_node"]  # whole nodes: nobody else's job on them during timing
        else:
            per_node = max(P["min_ncpus"], -(-max(r["ranks"] * r["threads"] for r in runs) // nodes))
        lines = []
        for r in runs:
            log = rel_logs / f"{r['name']}.log"
            lines.append(f'echo "$(date +%T) {r["name"]}"\n'
                         f'( {r["command"]} ) > {log} 2>&1 || echo "FAILED {r["name"]} (see {log})"')
        name = f"{args.exp}_n{nodes}"
        # free queue while the job fits its per-user limit, else the project's allocation
        queue_line = P["queue_line"]
        if nodes * per_node > P["free_max_cores"] or (args.charged and args.project):
            queue_line = f"#PBS -P {args.project}"
        # Vanda needs a memory request: the whole node for timing jobs, ~6 GB per core otherwise
        mem = P["mem_per_node"]
        if mem and not exclusive:
            mem = f":mem={max(8, 6 * per_node)}gb"
        select, modules = f"{nodes}:ncpus={per_node}:mpiprocs={per_node}{mem}", P["modules"]
        if gpu:
            select, modules = P["gpu_select"], P["gpu_modules"]
            queue_line = f"#PBS -P {args.project}" if args.charged and args.project else P["queue_line"]
        script = template.safe_substitute(SELECT=select,
            NAME=name, QUEUE_LINE=queue_line, MEM=mem, NODES=nodes, NCPUS=per_node, MPIPROCS=per_node,
            WALLTIME=args.walltime or walltime, PLACE="scatter:excl" if exclusive else "scatter",
            LOGDIR=rel_logs, MODULES=modules, COMMANDS="\n".join(lines),
        )
        path = out_dir / f"{name}.pbs"
        path.write_text(script)
        print(f"  {len(runs)} runs -> qsub {path.relative_to(ROOT)}   [{queue_line[5:]}, {nodes * per_node} cores]")


if __name__ == "__main__":
    main()
