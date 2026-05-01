// MPI-parallel symmetric eigenvalue example using arnoldi::Arnoldi + MPIComm.
//
// Problem: 1D discrete Laplacian of global size N = 100, evenly split across
// MPI ranks.  A = tridiag(-1, 2, -1).  We find the 4 smallest eigenvalues.
//
// Build:
//   mpicxx -std=c++17 -O2 arnoldi_mpi.cpp -o arnoldi_mpi \
//          -I../include -llapack -lblas
// Run:
//   mpirun -np 4 ./arnoldi_mpi

#include <mpi.h>
#include <arnoldi/arnoldi.hpp>
#include <arnoldi/mpi.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = 100;
    if (N % nprocs != 0) {
        if (rank == 0)
            std::printf("N=%d must be divisible by nprocs=%d\n", N, nprocs);
        MPI_Finalize();
        return 1;
    }

    const int n_local = N / nprocs;
    const int global_offset = rank * n_local;

    // Distributed matvec: y = A * x, where A = tridiag(-1, 2, -1).
    // Each rank owns rows [global_offset, global_offset + n_local).
    // Boundary elements require one-element halo exchange with neighbours.
    auto matvec = [&](const double* x, double* y) {
        double left_ghost = 0.0, right_ghost = 0.0;

        MPI_Request reqs[4];
        int nreq = 0;

        if (rank > 0) {
            MPI_Isend(&x[0], 1, MPI_DOUBLE, rank - 1, 0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(&left_ghost, 1, MPI_DOUBLE, rank - 1, 1,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (rank < nprocs - 1) {
            MPI_Isend(&x[n_local - 1], 1, MPI_DOUBLE, rank + 1, 1,
                      MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(&right_ghost, 1, MPI_DOUBLE, rank + 1, 0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }

        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        for (int i = 0; i < n_local; ++i) {
            double val = 2.0 * x[i];
            if (i > 0)                             val -= x[i - 1];
            else if (global_offset > 0)            val -= left_ghost;
            if (i < n_local - 1)                   val -= x[i + 1];
            else if (global_offset + n_local < N)  val -= right_ghost;
            y[i] = val;
        }
    };

    const int nev = 4;
    const int ncv = 20;

    arnoldi::MPIComm comm(MPI_COMM_WORLD);
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double, arnoldi::MPIComm>
        solver("I", n_local, "SM", nev, ncv, comm);
    solver.tol(0.0).maxiter(1000).mode(1).ishift(1);

    solver.solve(matvec);

    if (solver.info() < 0) {
        if (rank == 0)
            std::printf("Error: saupd returned info = %d\n", solver.info());
        MPI_Finalize();
        return 1;
    }

    auto r = solver.eigenpairs(true, 0.0);
    const int nconv = solver.num_converged();

    if (rank == 0) {
        std::printf("\n Ritz values and relative residuals\n");
        std::printf(" ----------------------------------\n");
    }
    std::vector<double> ax(n_local);
    for (int j = 0; j < nconv; ++j) {
        const double* xj = &r.vectors[j * n_local];
        matvec(xj, ax.data());

        double local_sq = 0.0;
        for (int k = 0; k < n_local; ++k) {
            double t = ax[k] - r.values[j] * xj[k];
            local_sq += t * t;
        }
        double global_sq;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        double nrm = std::sqrt(global_sq) / std::abs(r.values[j]);
        if (rank == 0)
            std::printf(" Row%4d:  lambda = %19.11e  resid = %e\n",
                        j + 1, r.values[j], nrm);
    }

    // Exact eigenvalues: 2 - 2*cos(k*pi/(N+1)), k = 1..N
    if (rank == 0) {
        std::printf("\n Comparison with exact eigenvalues\n");
        std::printf(" ---------------------------------\n");
        for (int j = 0; j < nconv; ++j) {
            double exact = 2.0 - 2.0 * std::cos((j + 1) * M_PI / (N + 1));
            std::printf("   computed = %19.11e   exact = %19.11e   diff = %e\n",
                        r.values[j], exact, std::abs(r.values[j] - exact));
        }

        std::printf("\n Summary\n -------\n");
        std::printf(" Global matrix size N = %d, ranks = %d, n_local = %d\n",
                    N, nprocs, n_local);
        std::printf(" Requested nev = %d, ncv = %d\n", nev, ncv);
        std::printf(" Converged Ritz values: %d\n", nconv);
        std::printf(" Arnoldi iterations:    %d\n", solver.num_iterations());
        std::printf(" OP*x applications:     %d\n", solver.num_op_applies());
    }

    MPI_Finalize();
    return 0;
}
