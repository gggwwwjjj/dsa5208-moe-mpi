#include "alltoall.hpp"

#include <algorithm>
#include <stdexcept>

#include "allreduce.hpp"  // mpi_real

namespace {

// Exchanges the counts if needed, sizes recv, and returns element displacements.
void prepare(const rvec& send, const std::vector<int>& send_counts, int row_len,
             rvec& recv, std::vector<int>& recv_counts, bool counts_known, MPI_Comm comm,
             std::vector<int>& scount, std::vector<int>& sdispl, std::vector<int>& rcount,
             std::vector<int>& rdispl) {
    int P = 1;
    MPI_Comm_size(comm, &P);
    if (static_cast<int>(send_counts.size()) != P) throw std::runtime_error("alltoall: send_counts size != ranks");
    if (!counts_known) {
        recv_counts.assign(static_cast<std::size_t>(P), 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    } else if (static_cast<int>(recv_counts.size()) != P) {
        throw std::runtime_error("alltoall: known recv_counts size != ranks");
    }
    scount.resize(static_cast<std::size_t>(P));
    rcount.resize(static_cast<std::size_t>(P));
    sdispl.assign(static_cast<std::size_t>(P) + 1, 0);
    rdispl.assign(static_cast<std::size_t>(P) + 1, 0);
    for (int r = 0; r < P; ++r) {
        const std::size_t i = static_cast<std::size_t>(r);
        scount[i] = send_counts[i] * row_len;
        rcount[i] = recv_counts[i] * row_len;
        sdispl[i + 1] = sdispl[i] + scount[i];
        rdispl[i + 1] = rdispl[i] + rcount[i];
    }
    if (static_cast<std::size_t>(sdispl.back()) != send.size()) throw std::runtime_error("alltoall: send size != counts");
    recv.resize(static_cast<std::size_t>(rdispl.back()));
}

}  // namespace

void alltoall_mpi(const rvec& send, const std::vector<int>& send_counts, int row_len,
                  rvec& recv, std::vector<int>& recv_counts, bool counts_known, MPI_Comm comm) {
    std::vector<int> scount, sdispl, rcount, rdispl;
    prepare(send, send_counts, row_len, recv, recv_counts, counts_known, comm, scount, sdispl, rcount, rdispl);
    MPI_Alltoallv(send.data(), scount.data(), sdispl.data(), mpi_real(), recv.data(), rcount.data(), rdispl.data(),
                  mpi_real(), comm);
}

void alltoall_pairwise(const rvec& send, const std::vector<int>& send_counts, int row_len,
                       rvec& recv, std::vector<int>& recv_counts, bool counts_known, MPI_Comm comm) {
    std::vector<int> scount, sdispl, rcount, rdispl;
    prepare(send, send_counts, row_len, recv, recv_counts, counts_known, comm, scount, sdispl, rcount, rdispl);
    int P = 1, me = 0;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &me);
    std::vector<MPI_Request> req;
    req.reserve(2 * static_cast<std::size_t>(P));
    // receives first, so every message finds a posted buffer; peers are visited
    // starting after `me`, so the P ranks do not all target rank 0 at once
    for (int s = 1; s < P; ++s) {
        const int src = (me - s + P) % P;
        const std::size_t i = static_cast<std::size_t>(src);
        if (rcount[i] == 0) continue;
        req.emplace_back();
        MPI_Irecv(recv.data() + rdispl[i], rcount[i], mpi_real(), src, 7, comm, &req.back());
    }
    for (int s = 1; s < P; ++s) {
        const int dst = (me + s) % P;
        const std::size_t i = static_cast<std::size_t>(dst);
        if (scount[i] == 0) continue;
        req.emplace_back();
        MPI_Isend(send.data() + sdispl[i], scount[i], mpi_real(), dst, 7, comm, &req.back());
    }
    const std::size_t m = static_cast<std::size_t>(me);
    std::copy(send.begin() + sdispl[m], send.begin() + sdispl[m] + scount[m], recv.begin() + rdispl[m]);
    MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
}

void alltoall(const std::string& variant, const rvec& send, const std::vector<int>& send_counts,
              int row_len, rvec& recv, std::vector<int>& recv_counts, bool counts_known, MPI_Comm comm) {
    if (variant == "mpi") alltoall_mpi(send, send_counts, row_len, recv, recv_counts, counts_known, comm);
    else if (variant == "pairwise") alltoall_pairwise(send, send_counts, row_len, recv, recv_counts, counts_known, comm);
    else throw std::runtime_error("unknown all-to-all variant " + variant);
}
