// C++ port of EXAMPLES/SYM/dsdrv2.f using arnoldi::Arnoldi<Kind::Sym, double>.
//
// Symmetric eigenvalue problem in shift-invert mode (Mode 3):
//
//     A*x = lambda*x,   OP = (A - sigma*I)^{-1},   B = I
//
// where A is the 1-D Dirichlet Laplacian on [0,1], scaled by 1/h^2.
// WHICH="LM" finds the largest-magnitude eigenvalues of OP, which correspond
// to eigenvalues of A closest to sigma.  After extraction, dseupd maps the
// Ritz values of OP back to those of A.
//
// The linear system (A - sigma*I)*y = x is solved with LAPACK's tridiagonal
// factor (dgttrf) + solve (dgttrs).

#include <arnoldi/arnoldi.hpp>
#include "lapack_extra.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// A*x for the 1-D Laplacian (used only for residual check, not OP).
static void av(int n, const double* v, double* w) {
    const double inv_h2 = static_cast<double>((n + 1) * (n + 1));
    w[0] = 2.0 * v[0] - v[1];
    for (int j = 1; j < n - 1; ++j)
        w[j] = -v[j - 1] + 2.0 * v[j] - v[j + 1];
    w[n - 1] = -v[n - 2] + 2.0 * v[n - 1];
    for (int j = 0; j < n; ++j) w[j] *= inv_h2;
}

int main() {
    const int    n     = 100;
    const int    nev   = 4;
    const int    ncv   = 10;
    const double sigma = 0.0;

    // Build and factor (A - sigma*I) as a tridiagonal.
    const double h2 = 1.0 / ((n + 1.0) * (n + 1.0));
    std::vector<double> ad(n, 2.0 / h2 - sigma);
    std::vector<double> adl(n, -1.0 / h2);
    std::vector<double> adu(n, -1.0 / h2);
    std::vector<double> adu2(n, 0.0);
    std::vector<int>    ipiv(n);

    if (int info = arnoldi_examples::lapackx::gttrf<double>(
            n, adl.data(), ad.data(), adu.data(), adu2.data(), ipiv.data()); info != 0) {
        std::printf(" Error with _gttrf in _SDRV2.  info=%d\n", info);
        return 1;
    }

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> solver("I", n, "LM", nev, ncv);
    solver.tol(0.0).maxiter(300).mode(3).ishift(1);

    // OP: y = (A - sigma*I)^{-1} * x
    solver.solve([&](const double* x, double* y) {
        std::copy(x, x + n, y);
        arnoldi_examples::lapackx::gttrs<double>("N", n, 1,
                                                adl.data(), ad.data(), adu.data(),
                                                adu2.data(), ipiv.data(), y, n);
    });

    if (solver.info() < 0) {
        std::printf(" Error with _saupd, info = %d\n", solver.info());
        return 1;
    }

    // seupd maps Ritz values of OP back to eigenvalues of A using sigma.
    auto r = solver.eigenpairs(true, sigma);
    const int nconv = solver.num_converged();

    // Relative residuals using true A (not OP).
    std::vector<double> ax(n);
    std::printf("\n Ritz values and relative residuals\n");
    std::printf(" ----------------------------------\n");
    for (int j = 0; j < nconv; ++j) {
        const double* xj = &r.vectors[j * n];
        av(n, xj, ax.data());
        double s = 0.0;
        for (int k = 0; k < n; ++k) {
            double t = ax[k] - r.values[j] * xj[k];
            s += t * t;
        }
        double nrm = std::sqrt(s) / std::abs(r.values[j]);
        std::printf(" Row%4d: %19.11e %19.11e\n", j + 1, r.values[j], nrm);
    }

    std::printf("\n _SDRV2 \n ====== \n");
    std::printf(" Size of the matrix is                  %d\n", n);
    std::printf(" The number of Ritz values requested    %d\n", nev);
    std::printf(" The number of Arnoldi vectors (NCV)    %d\n", ncv);
    std::printf(" What portion of the spectrum:          LM (closest to sigma=%.3f)\n", sigma);
    std::printf(" The number of converged Ritz values    %d\n", nconv);
    std::printf(" Number of IA update iterations         %d\n", solver.num_iterations());
    std::printf(" The number of OP*x                     %d\n", solver.num_op_applies());
    return 0;
}
