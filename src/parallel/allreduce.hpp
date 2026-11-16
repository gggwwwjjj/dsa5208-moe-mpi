#pragma once
// Gradient all-reduce variants (plan D8). Both sum `buf` in place over `comm`.
#include <mpi.h>

#include <cstddef>
#include <string>

#include "../core/config.hpp"

inline MPI_Datatype mpi_real() { return sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT; }

// (a) library collective
void allreduce_mpi(real* buf, std::size_t n, MPI_Comm comm);

// (b) ring all-reduce built on MPI_Sendrecv (L7 p42-44): reduce-scatter in
// P-1 steps, then all-gather in P-1 steps. Each rank sends 2(P-1)/P * n values.
void allreduce_ring(real* buf, std::size_t n, MPI_Comm comm);

// Select by name ("mpi" or "ring").
void allreduce(const std::string& variant, real* buf, std::size_t n, MPI_Comm comm);

// ZeRO-1 building blocks (plan D9): an all-reduce split in two, with the
// optimiser step on each rank's shard in between.
//   zero_shard:      the part [lo, hi) of an n-element buffer this rank owns
//   reduce_scatter:  afterwards buf[lo, hi) holds the sum over all ranks
//                    (the rest of buf is scratch)
//   all_gather:      every rank's buf[lo, hi) is copied to all ranks
// "mpi": MPI_Reduce_scatter + MPI_Allgatherv, rank r owns chunk r.
// "ring": the two halves of allreduce_ring, rank r owns chunk (r + 1) mod P,
// so ring all-reduce and ring ZeRO-1 add the numbers in the same order.
struct Shard {
    std::size_t lo = 0, hi = 0;
};
Shard zero_shard(const std::string& variant, std::size_t n, MPI_Comm comm);
void reduce_scatter(const std::string& variant, real* buf, std::size_t n, MPI_Comm comm);
void all_gather(const std::string& variant, real* buf, std::size_t n, MPI_Comm comm);
