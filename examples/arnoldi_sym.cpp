// arnoldi_sym — smallest eigenvalues of the 1D Dirichlet Laplacian
// via arnoldi::Arnoldi<Kind::Sym, double>.

#include <arnoldi/arnoldi.hpp>

#include <cstdio>

int main() {
    const int n   = 128;
    const int nev = 5;
    const int ncv = 2 * nev + 1;

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> solver("I", n, "SM", nev, ncv);
    solver.tol(0.0).maxiter(300);

    solver.solve([&](const double* x, double* y) {
        y[0] = 2.0 * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i)
            y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
        y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
    });

    if (!solver.converged()) {
        std::printf("saupd failed, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs();
    std::printf("Smallest %d eigenvalues (1D Laplacian, n=%d):\n", nev, n);
    for (int i = 0; i < nev; ++i)
        std::printf("  lambda[%d] = %.12f\n", i, r.values[i]);
    std::printf("iterations=%d, OP applies=%d\n",
                solver.num_iterations(), solver.num_op_applies());
    return 0;
}
