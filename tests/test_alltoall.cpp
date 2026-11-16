// Token exchange (plan §6): both variants deliver every row to the right rank
// in source-rank order, with uneven and zero counts, with counts exchanged or
// given (counts_known), and the reverse exchange returns the original rows.
#include <mpi.h>

#include <cstdio>
#include <vector>

#include "../src/parallel/alltoall.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int row_len = 3;  // [source, destination, index]
    // uneven counts, including zeros: rank s sends (s + 2*d + trial) % 4 rows to d
    auto count = [](int s, int d, int trial) { return (s + 2 * d + trial) % 4; };
    int failures = 0;
    for (const char* variant : {"mpi", "pairwise"}) {
        for (int trial = 0; trial < 3; ++trial) {
            std::vector<int> send_counts(static_cast<std::size_t>(size));
            rvec send;
            for (int d = 0; d < size; ++d) {
                send_counts[static_cast<std::size_t>(d)] = count(rank, d, trial);
                for (int i = 0; i < count(rank, d, trial); ++i)
                    send.insert(send.end(), {real(rank), real(d), real(i)});
            }
            rvec recv;
            std::vector<int> recv_counts;
            alltoall(variant, send, send_counts, row_len, recv, recv_counts, false, MPI_COMM_WORLD);
            // expected: rows from source 0, 1, ... in order, each addressed to me
            std::size_t at = 0;
            bool ok = true;
            for (int s = 0; s < size; ++s) {
                ok = ok && recv_counts[static_cast<std::size_t>(s)] == count(s, rank, trial);
                for (int i = 0; i < count(s, rank, trial); ++i, at += row_len)
                    ok = ok && at + 2 < recv.size() && recv[at] == real(s) && recv[at + 1] == real(rank) &&
                         recv[at + 2] == real(i);
            }
            ok = ok && at == recv.size();
            // reverse exchange with known counts gives back exactly what was sent
            rvec back;
            std::vector<int> back_counts = send_counts;
            alltoall(variant, recv, recv_counts, row_len, back, back_counts, true, MPI_COMM_WORLD);
            ok = ok && back == send;
            int bad = ok ? 0 : 1;
            MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (bad) {
                ++failures;
                if (rank == 0) std::printf("  %s trial %d: %d ranks wrong\n", variant, trial, bad);
            }
        }
    }
    if (rank == 0) std::printf("test_alltoall [%d ranks]: %s\n", size, failures == 0 ? "PASS" : "FAIL");
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
