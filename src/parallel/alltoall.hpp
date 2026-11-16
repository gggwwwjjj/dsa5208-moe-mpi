#pragma once
// Token exchange for expert parallelism (plan D8 (c), (d), §5.3). Every rank
// sends send_counts[r] rows of row_len values to rank r of `comm` (rows already
// ordered by destination) and receives recv_counts[r] rows from rank r, stored
// in source-rank order.
//
// The receiver must know recv_counts before the payload arrives. For the first
// exchange of a step (dispatch) they come from an MPI_Alltoall of the counts;
// the other three exchanges of a MoE layer reuse counts already known
// (combine sends back what dispatch received), so counts_known skips it.
#include <mpi.h>

#include <string>
#include <vector>

#include "../core/memory.hpp"

// (c) MPI_Alltoallv
void alltoall_mpi(const rvec& send, const std::vector<int>& send_counts, int row_len,
                  rvec& recv, std::vector<int>& recv_counts, bool counts_known, MPI_Comm comm);

// (d) own pairwise exchange: Irecv from every peer, then Isend to every peer,
// one Waitall; the local block is copied. send and recv are separate buffers and
// neither is touched before Waitall (L7 p44).
void alltoall_pairwise(const rvec& send, const std::vector<int>& send_counts, int row_len,
                       rvec& recv, std::vector<int>& recv_counts, bool counts_known, MPI_Comm comm);

// Select by name ("mpi" or "pairwise").
void alltoall(const std::string& variant, const rvec& send, const std::vector<int>& send_counts,
              int row_len, rvec& recv, std::vector<int>& recv_counts, bool counts_known, MPI_Comm comm);
