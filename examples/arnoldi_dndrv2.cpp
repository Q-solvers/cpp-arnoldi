// C++ port of EXAMPLES/NONSYM/dndrv2.f using arnoldi::Arnoldi<Kind::Nonsym, double>.
//
// Nonsymmetric eigenvalue problem in shift-invert mode (Mode 3, real shift):
//
//     A*x = lambda*x,   OP = (A - sigmar*I)^{-1},   B = I
//
// A is the 1-D convection-diffusion operator  u" + rho*u'  on [0,1] with
// zero Dirichlet boundary conditions, discretised by central differences.
//
//     dl = -1 - s,   dd = 2,   du = -1 + s,      where s = rho*h/2
//
// WHICH="LM" finds the largest-magnitude eigenvalues of OP → those of A
// closest to sigmar.  dneupd maps Ritz values back using (sigmar, sigmai=0).

#include <arnoldi/arnoldi.hpp>
#include "lapack_extra.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static constexpr double RHO = 10.0;

// A*x for the 1-D convection-diffusion operator (for residual check).
static void av(int n, const double* v, double* w) {
    const double h  = 1.0 / (n + 1);
    const double s  = RHO * h / 2.0;
    const double dl = -1.0 - s;
    const double dd =  2.0;
    const double du = -1.0 + s;

    w[0] = dd * v[0] + du * v[1];
    for (int j = 1; j < n - 1; ++j)
        w[j] = dl * v[j - 1] + dd * v[j] + du * v[j + 1];
    w[n - 1] = dl * v[n - 2] + dd * v[n - 1];
}

int main() {
    const int    n      = 100;
    const int    nev    = 4;
    const int    ncv    = 20;
    const double sigmar = 1.0;
    const double sigmai = 0.0;

    // Build and factor (A - sigmar*I) as a tridiagonal.
    const double h  = 1.0 / (n + 1);
    const double s  = RHO * h / 2.0;
    const double s1 = -1.0 - s;
    const double s2 =  2.0 - sigmar;
    const double s3 = -1.0 + s;

    std::vector<double> dd(n, s2);
    std::vector<double> dl(n, s1);
    std::vector<double> du(n, s3);
    std::vector<double> du2(n, 0.0);
    std::vector<int>    ipiv(n);

    if (int info = arnoldi_examples::lapackx::gttrf<double>(
            n, dl.data(), dd.data(), du.data(), du2.data(), ipiv.data()); info != 0) {
        std::printf(" Error with _gttrf in _NDRV2.  info=%d\n", info);
        return 1;
    }

    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> solver("I", n, "LM", nev, ncv);
    solver.tol(0.0).maxiter(300).mode(3).ishift(1);

    // OP: y = (A - sigmar*I)^{-1} * x
    solver.solve([&](const double* x, double* y) {
        std::copy(x, x + n, y);
        arnoldi_examples::lapackx::gttrs<double>("N", n, 1,
                                                dl.data(), dd.data(), du.data(),
                                                du2.data(), ipiv.data(), y, n);
    });

    if (solver.info() < 0) {
        std::printf(" Error with _naupd, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs(true, sigmar, sigmai);
    const int nconv = solver.num_converged();

    // Residuals: for a real Ritz value use xj directly; for a conjugate pair
    // take xj + i*x_{j+1} and check (A*v − λ*v) in both parts.
    std::vector<double> ax(n), axi(n);
    std::printf("\n Ritz values (Real, Imag) and relative residuals\n");
    std::printf(" -----------------------------------------------\n");
    for (int j = 0; j < nconv; ) {
        const double lr = r.values_re[j];
        const double li = r.values_im[j];
        const double* xj = &r.vectors[j * n];
        if (li == 0.0) {
            av(n, xj, ax.data());
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                double t = ax[k] - lr * xj[k];
                sum += t * t;
            }
            double nrm = std::sqrt(sum) / std::abs(lr);
            std::printf(" Row%4d: %19.11e %19.11e %19.11e\n", j + 1, lr, li, nrm);
            ++j;
        } else {
            const double* xi = &r.vectors[(j + 1) * n];
            av(n, xj, ax.data());
            av(n, xi, axi.data());
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                double re = ax[k]  - (lr * xj[k] - li * xi[k]);
                double im = axi[k] - (li * xj[k] + lr * xi[k]);
                sum += re * re + im * im;
            }
            double nrm = std::sqrt(sum) / std::hypot(lr, li);
            std::printf(" Row%4d: %19.11e %19.11e %19.11e\n", j + 1, lr,  li, nrm);
            std::printf(" Row%4d: %19.11e %19.11e %19.11e\n", j + 2, lr, -li, nrm);
            j += 2;
        }
    }

    std::printf("\n _NDRV2 \n ====== \n");
    std::printf(" Size of the matrix is                  %d\n", n);
    std::printf(" The number of Ritz values requested    %d\n", nev);
    std::printf(" The number of Arnoldi vectors (NCV)    %d\n", ncv);
    std::printf(" What portion of the spectrum:          LM (closest to sigma=%.3f)\n", sigmar);
    std::printf(" The number of converged Ritz values    %d\n", nconv);
    std::printf(" Number of IA update iterations         %d\n", solver.num_iterations());
    std::printf(" The number of OP*x                     %d\n", solver.num_op_applies());
    return 0;
}
