#ifndef ARNOLDI_MPI_HPP
#define ARNOLDI_MPI_HPP

// MPI-aware communicator for distributed ARPACK solves.
//
// Usage:
//   #include <mpi.h>
//   #include <arnoldi/arnoldi.hpp>
//   #include <arnoldi/mpi.hpp>
//
//   arnoldi::MPIComm comm(MPI_COMM_WORLD);
//   arnoldi::Arnoldi<arnoldi::Kind::Sym, double, arnoldi::MPIComm>
//       solver("I", n_local, "LM", nev, ncv, comm);
//   solver.solve([&](const double* x, double* y) { /* local matvec */ });
//
// Each MPI rank holds its local portion of the distributed vectors
// (n = n_local). The communicator handles MPI_Allreduce for dot
// products, norms, and Gram–Schmidt projections.

#include <mpi.h>

#include <complex>

namespace arnoldi {

  struct MPIComm {
    MPI_Comm mpi_comm;

    explicit MPIComm(MPI_Comm c = MPI_COMM_WORLD) noexcept : mpi_comm(c) {}

    int allreduce_sum(int v) const {
      int out;
      MPI_Allreduce(&v, &out, 1, MPI_INT, MPI_SUM, mpi_comm);
      return out;
    }
    float allreduce_sum(float v) const {
      float out;
      MPI_Allreduce(&v, &out, 1, MPI_FLOAT, MPI_SUM, mpi_comm);
      return out;
    }
    double allreduce_sum(double v) const {
      double out;
      MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
      return out;
    }

    void allreduce_sum(float* data, int count) const { MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, mpi_comm); }
    void allreduce_sum(double* data, int count) const { MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, mpi_comm); }
    void allreduce_sum(std::complex<float>* data, int count) const {
      MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_CXX_FLOAT_COMPLEX, MPI_SUM, mpi_comm);
    }
    void allreduce_sum(std::complex<double>* data, int count) const {
      MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, mpi_comm);
    }
  };

}  // namespace arnoldi

#endif  // ARNOLDI_MPI_HPP
