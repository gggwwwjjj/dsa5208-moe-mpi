#include "allreduce.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

struct Ring {
    int P = 1, r = 0;
    std::size_t n = 0;
    Ring(std::size_t n_, MPI_Comm comm) : n(n_) {
        MPI_Comm_size(comm, &P);
        MPI_Comm_rank(comm, &r);
    }
    // chunk c covers [lo(c), lo(c+1)); sizes differ by at most one when P does not divide n
    std::size_t lo(int c) const { return n * static_cast<std::size_t>(c) / static_cast<std::size_t>(P); }
    int len(int c) const { return static_cast<int>(lo(c + 1) - lo(c)); }
    int right() const { return (r + 1) % P; }
    int left() const { return (r - 1 + P) % P; }
};

// Reduce-scatter: at step s rank r sends chunk (r - s) and adds the chunk
// (r - s - 1) received from its left neighbour. After P-1 steps rank r
// holds the complete sum of chunk (r + 1) mod P.
void ring_reduce_scatter(real* buf, std::size_t n, MPI_Comm comm) {
    const Ring g(n, comm);
    if (g.P == 1) return;
    std::vector<real> incoming(n / static_cast<std::size_t>(g.P) + 1);
    for (int s = 0; s < g.P - 1; ++s) {
        const int send_c = (g.r - s + g.P) % g.P, recv_c = (g.r - s - 1 + g.P) % g.P;
        MPI_Sendrecv(buf + g.lo(send_c), g.len(send_c), mpi_real(), g.right(), 0,
                     incoming.data(), g.len(recv_c), mpi_real(), g.left(), 0, comm, MPI_STATUS_IGNORE);
        real* dst = buf + g.lo(recv_c);
        for (int i = 0; i < g.len(recv_c); ++i) dst[i] += incoming[static_cast<std::size_t>(i)];
    }
}

// All-gather: pass the completed chunks around the ring, overwriting.
void ring_all_gather(real* buf, std::size_t n, MPI_Comm comm) {
    const Ring g(n, comm);
    for (int s = 0; s < g.P - 1; ++s) {
        const int send_c = (g.r + 1 - s + g.P) % g.P, recv_c = (g.r - s + g.P) % g.P;
        MPI_Sendrecv(buf + g.lo(send_c), g.len(send_c), mpi_real(), g.right(), 1,
                     buf + g.lo(recv_c), g.len(recv_c), mpi_real(), g.left(), 1, comm, MPI_STATUS_IGNORE);
    }
}

// element counts and displacements of the MPI variant (chunk r for rank r)
void mpi_chunks(std::size_t n, MPI_Comm comm, std::vector<int>& counts, std::vector<int>& displs) {
    const Ring g(n, comm);
    counts.resize(static_cast<std::size_t>(g.P));
    displs.resize(static_cast<std::size_t>(g.P));
    for (int c = 0; c < g.P; ++c) {
        counts[static_cast<std::size_t>(c)] = g.len(c);
        displs[static_cast<std::size_t>(c)] = static_cast<int>(g.lo(c));
    }
}

}  // namespace

void allreduce_mpi(real* buf, std::size_t n, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), mpi_real(), MPI_SUM, comm);
}

void allreduce_ring(real* buf, std::size_t n, MPI_Comm comm) {
    ring_reduce_scatter(buf, n, comm);
    ring_all_gather(buf, n, comm);
}

void allreduce(const std::string& variant, real* buf, std::size_t n, MPI_Comm comm) {
    if (variant == "mpi") allreduce_mpi(buf, n, comm);
    else if (variant == "ring") allreduce_ring(buf, n, comm);
    else throw std::runtime_error("unknown all-reduce variant " + variant);
}

Shard zero_shard(const std::string& variant, std::size_t n, MPI_Comm comm) {
    const Ring g(n, comm);
    const int c = variant == "ring" ? (g.r + 1) % g.P : g.r;
    return {g.lo(c), g.lo(c + 1)};
}

void reduce_scatter(const std::string& variant, real* buf, std::size_t n, MPI_Comm comm) {
    if (variant == "ring") return ring_reduce_scatter(buf, n, comm);
    if (variant != "mpi") throw std::runtime_error("unknown all-reduce variant " + variant);
    std::vector<int> counts, displs;
    mpi_chunks(n, comm, counts, displs);
    int r = 0;
    MPI_Comm_rank(comm, &r);
    std::vector<real> mine(static_cast<std::size_t>(counts[static_cast<std::size_t>(r)]));
    MPI_Reduce_scatter(buf, mine.data(), counts.data(), mpi_real(), MPI_SUM, comm);
    std::copy(mine.begin(), mine.end(), buf + displs[static_cast<std::size_t>(r)]);
}

void all_gather(const std::string& variant, real* buf, std::size_t n, MPI_Comm comm) {
    if (variant == "ring") return ring_all_gather(buf, n, comm);
    if (variant != "mpi") throw std::runtime_error("unknown all-reduce variant " + variant);
    std::vector<int> counts, displs;
    mpi_chunks(n, comm, counts, displs);
    MPI_Allgatherv(MPI_IN_PLACE, 0, mpi_real(), buf, counts.data(), displs.data(), mpi_real(), comm);
}
