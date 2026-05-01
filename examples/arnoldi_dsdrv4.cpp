// C++ port of EXAMPLES/SYM/dsdrv4.f using arnoldi::Arnoldi<Kind::Sym, double>.
//
// Generalized symmetric eigenvalue problem in shift-invert mode (Mode 3):
//
//     A*x = lambda*M*x,   OP = inv(A - sigma*M)*M,   B = M
//
// A is the 1-D FE Laplacian (stiffness matrix) on [0,1] with zero Dirichlet BC.
// M is the associated FE mass matrix (piecewise linear elements).
//
// WHICH="LM" finds the largest-magnitude eigenvalues of OP, corresponding to
// eigenvalues of (A, M) closest to sigma.

#include <arnoldi/arnoldi.hpp>
#include "lapack_extra.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static void av(int n, const double* v, double* w) {
    const double h = 1.0 / (n + 1);
    w[0] = (2.0 / h) * v[0] - (1.0 / h) * v[1];
    for (int j = 1; j < n - 1; ++j)
        w[j] = -(1.0 / h) * v[j - 1] + (2.0 / h) * v[j] - (1.0 / h) * v[j + 1];
    w[n - 1] = -(1.0 / h) * v[n - 2] + (2.0 / h) * v[n - 1];
}

static void mv(int n, const double* v, double* w) {
    const double h = 1.0 / (n + 1);
    w[0] = (4.0 * h / 6.0) * v[0] + (1.0 * h / 6.0) * v[1];
    for (int j = 1; j < n - 1; ++j)
        w[j] = (h / 6.0) * v[j - 1] + (4.0 * h / 6.0) * v[j] + (h / 6.0) * v[j + 1];
    w[n - 1] = (h / 6.0) * v[n - 2] + (4.0 * h / 6.0) * v[n - 1];
}

int main() {
    const int    n     = 100;
    const int    nev   = 4;
    const int    ncv   = 10;
    const double sigma = 0.0;

    const double h  = 1.0 / (n + 1);
    const double r1 = (4.0 / 6.0) * h;
    const double r2 = (1.0 / 6.0) * h;

    // Factor (A - sigma*M) as tridiagonal
    std::vector<double> ad(n), adl(n), adu(n), adu2(n, 0.0);
    std::vector<int>    ipiv(n);
    for (int j = 0; j < n; ++j) {
        ad[j]  = 2.0 / h - sigma * r1;
        adl[j] = -1.0 / h - sigma * r2;
    }
    std::copy(adl.begin(), adl.end(), adu.begin());

    if (int info = arnoldi_examples::lapackx::gttrf<double>(
            n, adl.data(), ad.data(), adu.data(), adu2.data(), ipiv.data()); info != 0) {
        std::printf(" Error with _gttrf in _SDRV4.  info=%d\n", info);
        return 1;
    }

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> solver("G", n, "LM", nev, ncv);
    solver.tol(0.0).maxiter(300).mode(3).ishift(1);

    // OP: y = inv(A - sigma*M) * M*x
    auto op = [&](const double* x, double* y) {
        mv(n, x, y);
        arnoldi_examples::lapackx::gttrs<double>("N", n, 1,
            adl.data(), ad.data(), adu.data(), adu2.data(), ipiv.data(), y, n);
    };

    // B: y = M*x
    auto bop = [&](const double* x, double* y) {
        mv(n, x, y);
    };

    solver.solve(op, bop);

    if (solver.info() < 0) {
        std::printf(" Error with _saupd, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs(true, sigma);
    const int nconv = solver.num_converged();

    // Residuals: ||A*x - lambda*M*x|| / |lambda|
    std::vector<double> ax(n), mx_buf(n);
    std::printf("\n Ritz values and relative residuals\n");
    std::printf(" ----------------------------------\n");
    for (int j = 0; j < nconv; ++j) {
        const double* xj = &r.vectors[j * n];
        av(n, xj, ax.data());
        mv(n, xj, mx_buf.data());
        double s = 0.0;
        for (int k = 0; k < n; ++k) {
            double t = ax[k] - r.values[j] * mx_buf[k];
            s += t * t;
        }
        double nrm = std::sqrt(s) / std::abs(r.values[j]);
        std::printf(" Row%4d: %19.11e %19.11e\n", j + 1, r.values[j], nrm);
    }

    std::printf("\n _SDRV4 \n ====== \n");
    std::printf(" Size of the matrix is                  %d\n", n);
    std::printf(" The number of Ritz values requested    %d\n", nev);
    std::printf(" The number of Arnoldi vectors (NCV)    %d\n", ncv);
    std::printf(" What portion of the spectrum:          LM (closest to sigma=%.3f)\n", sigma);
    std::printf(" The number of converged Ritz values    %d\n", nconv);
    std::printf(" Number of IA update iterations         %d\n", solver.num_iterations());
    std::printf(" The number of OP*x                     %d\n", solver.num_op_applies());
    return 0;
}
