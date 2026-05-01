#ifndef ARNOLDI_COMM_HPP
#define ARNOLDI_COMM_HPP

// Communicator abstraction for distributed (MPI) or serial ARPACK solves.
//
// SerialComm (default) — all reductions are identity; compiles to no-ops.
//
// Users can provide their own Comm type satisfying:
//   T    allreduce_sum(T local)          const;   // scalar reduce
//   void allreduce_sum(T* data, int n)   const;   // in-place array reduce
//
// See arpack/mpi.hpp for an MPI-aware implementation.

namespace arnoldi {

  struct SerialComm {
    template <typename T>
    T allreduce_sum(T v) const noexcept {
      return v;
    }

    template <typename T>
    void allreduce_sum(T*, int) const noexcept {}
  };

}  // namespace arnoldi

#endif  // ARNOLDI_COMM_HPP
