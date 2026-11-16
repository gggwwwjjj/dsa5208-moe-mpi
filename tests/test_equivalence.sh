#!/bin/bash
# Serial vs parallel: parameters after 20 SGD steps must agree to 1e-10 in
# double precision (plan §6, D12).
#   data parallel (--ep 1): dense model on synthetic data and Fashion-MNIST,
#     MoE top-1 / top-2 with capacity drops; both all-reduce variants; P = 1, 2, 4, 8
#   expert parallel: MoE with drops, both all-to-all variants; P = 2 .. 16,
#     including P > E (expert replicas summed over the expert-DP group)
# The serial run uses --groups D, D = P / ep capacity groups (D16).
# Usage: tests/test_equivalence.sh <build dir (double)> <mpirun command>
set -euo pipefail
BUILD=${1:-build/local-double}
MPIRUN=${2:-mpirun --oversubscribe}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
TOL=1e-10
fail=0

check() {  # label, parallel dump
    local out
    if out=$("$BUILD/compare_params" "$TMP/serial.bin" "$1.bin" $TOL); then echo "  $2: $out"
    else echo "  $2: $out"; fail=1; fi
}

serial() {  # options -> prints the drop fraction
    "$BUILD/mlp_serial" "$@" --dump "$TMP/serial.bin" --quiet | sed -n 's/.* dropped=\([0-9.]*\).*/\1/p'
}

dp_case() {  # name, options
    local name=$1; shift
    for p in 1 2 4 8; do
        local drop
        drop=$(serial "$@" --groups $p)
        for v in mpi ring; do
            OMP_NUM_THREADS=1 $MPIRUN -np $p "$BUILD/mlp_parallel" "$@" --ep 1 --allreduce $v \
                --dump "$TMP/p.bin" --quiet > /dev/null
            check "$TMP/p" "$name P=$p dp/$v (dropped $drop)"
        done
    done
}

ep_case() {  # name, ep ("max" = min(P, E)), ranks, options
    local name=$1 ep=$2 ranks=$3; shift 3
    local experts
    experts=$(echo "$@" | sed -n 's/.*--experts \([0-9]*\).*/\1/p')
    for p in $ranks; do
        local e=$ep
        [ "$e" = max ] && e=$(( p < experts ? p : experts ))
        local d=$(( p / e )) drop
        drop=$(serial "$@" --groups $d)
        for v in mpi:mpi pairwise:mpi mpi:ring; do
            OMP_NUM_THREADS=1 $MPIRUN -np $p "$BUILD/mlp_parallel" "$@" --ep $e --alltoall ${v%:*} \
                --allreduce ${v#*:} --dump "$TMP/p.bin" --quiet > /dev/null
            check "$TMP/p" "$name P=$p ep=$e D=$d a2a=${v%:*} ar=${v#*:} (dropped $drop)"
        done
    done
}

COMMON="--optim sgd --lr 0.05 --steps 20 --warmup 0 --batch 64 --hidden 32 --ffn 48 --blocks 2 --data synthetic"
dp_case synthetic $COMMON
# MoE, data parallel only (every rank holds all experts); capacity 1.0 so tokens are dropped
dp_case moe-top2-dp $COMMON --experts 4 --topk 2 --capacity 1.0 --aux 0.05
dp_case moe-top1-dp $COMMON --experts 4 --topk 1 --capacity 1.25 --aux 0.05
# expert parallelism: P <= E (one EP group) and P > E (D replicas of every expert)
ep_case moe-top2-ep "max" "2 4 8 16" $COMMON --experts 4 --topk 2 --capacity 1.0 --aux 0.05
ep_case moe-top1-ep "max" "2 4 8 16" $COMMON --experts 8 --topk 1 --capacity 1.25 --aux 0.05
# two experts per rank and two replicas at once
ep_case moe-top2-ep4 4 "8" $COMMON --experts 8 --topk 2 --capacity 1.0 --aux 0.05
# checkpoint (D17): 10 serial steps saved, then serial and EP (P = 8, ep = 4, D = 2)
# both continue from it for 10 steps; a wrong expert slice in --load would differ
CK="$COMMON --experts 8 --topk 2 --capacity 1.0 --aux 0.05"
CK=${CK/--steps 20/--steps 10}
"$BUILD/mlp_serial" $CK --groups 1 --dump "$TMP/ckpt.bin" --quiet > /dev/null
"$BUILD/mlp_serial" $CK --groups 2 --load "$TMP/ckpt.bin" --seed 7 --dump "$TMP/serial.bin" --quiet > /dev/null
OMP_NUM_THREADS=1 $MPIRUN -np 8 "$BUILD/mlp_parallel" $CK --ep 4 --load "$TMP/ckpt.bin" --seed 7 \
    --dump "$TMP/p.bin" --quiet > /dev/null
check "$TMP/p" "checkpoint load P=8 ep=4 D=2"
# round trip: load with lr 0 (other seed) and dump again must give the checkpoint back
OMP_NUM_THREADS=1 $MPIRUN -np 8 "$BUILD/mlp_parallel" ${CK/--lr 0.05/--lr 0} --ep 4 --load "$TMP/ckpt.bin" \
    --seed 7 --dump "$TMP/p.bin" --quiet > /dev/null
cp "$TMP/ckpt.bin" "$TMP/serial.bin"
check "$TMP/p" "checkpoint round trip P=8 ep=4"
if "$BUILD/mlp_serial" ${CK/--experts 8/--experts 4} --load "$TMP/ckpt.bin" --quiet > /dev/null 2>&1; then
    echo "  checkpoint of another model accepted: FAIL"; fail=1
else
    echo "  checkpoint of another model rejected: PASS"
fi
# ZeRO-1 (D9): reduce-scatter, update own shard, all-gather. SGD vs serial for
# data parallel (dense + experts sharded over all ranks) and EP with D = 2
# (experts sharded over the expert-DP group)
Z="$COMMON --experts 8 --topk 2 --capacity 1.0 --aux 0.05"
for v in mpi ring; do
    "$BUILD/mlp_serial" $Z --groups 4 --dump "$TMP/serial.bin" --quiet > /dev/null
    OMP_NUM_THREADS=1 $MPIRUN -np 4 "$BUILD/mlp_parallel" $Z --ep 1 --zero --allreduce $v --dump "$TMP/p.bin" --quiet > /dev/null
    check "$TMP/p" "zero-1 dp P=4 ar=$v"
    "$BUILD/mlp_serial" $Z --groups 2 --dump "$TMP/serial.bin" --quiet > /dev/null
    OMP_NUM_THREADS=1 $MPIRUN -np 8 "$BUILD/mlp_parallel" $Z --ep 4 --zero --allreduce $v --dump "$TMP/p.bin" --quiet > /dev/null
    check "$TMP/p" "zero-1 ep=4 D=2 P=8 ar=$v"
done
# Adam: the ring all-reduce is reduce-scatter + all-gather, so ring with and
# without ZeRO-1 add in the same order and must agree bit for bit
ZA=${Z/--optim sgd --lr 0.05/--optim adam --lr 0.001}
for args in "-np 4 --ep 1" "-np 8 --ep 4"; do
    np=${args% --ep*}; ep=${args#*--ep }
    OMP_NUM_THREADS=1 $MPIRUN $np "$BUILD/mlp_parallel" $ZA --ep $ep --allreduce ring --dump "$TMP/serial.bin" --quiet > /dev/null
    OMP_NUM_THREADS=1 $MPIRUN $np "$BUILD/mlp_parallel" $ZA --ep $ep --allreduce ring --zero --dump "$TMP/p.bin" --quiet > /dev/null
    if out=$("$BUILD/compare_params" "$TMP/serial.bin" "$TMP/p.bin" 0); then echo "  zero-1 adam bitwise ($args): $out"
    else echo "  zero-1 adam bitwise ($args): $out"; fail=1; fi
done
if [ -f data/fashion_mnist/train_images.f32 ]; then
    dp_case fashion ${COMMON/--data synthetic/--data fashion} --no-eval
fi
if [ $fail -eq 0 ]; then echo "test_equivalence: PASS"; else echo "test_equivalence: FAIL"; exit 1; fi
