// C++ port of EXAMPLES/SYM/dsdrv1.f using arnoldi::Arnoldi<Kind::Sym, double>.
//
// Standard symmetric eigenvalue problem  A*x = lambda*x  (regular mode, Mode 1).
// A is the 2D Dirichlet Laplacian on the unit square, nx interior points per side.
//
//             | T -I          |
//             |-I  T -I       |
//    A  =  h² |   -I  T       |      T = tridiag(-1, 4, -1) / h²
//             |        ...  -I|      (h = 1/(nx+1), 1/h² applied globally)
//             |           -I T|
//
// Expected behaviour: "SM" gives the smallest-magnitude eigenvalues,
// which are the usual 2*(2 - cos(iπ/(nx+1)) - cos(jπ/(nx+1))) / h² mesh modes.

#include <arnoldi/arnoldi.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

static void tv(int nx, const double* x, double* y) {
    constexpr double dd = 4.0, dl = -1.0, du = -1.0;
    y[0] = dd * x[0] + du * x[1];
    for (int j = 1; j < nx - 1; ++j)
        y[j] = dl * x[j - 1] + dd * x[j] + du * x[j + 1];
    y[nx - 1] = dl * x[nx - 2] + dd * x[nx - 1];
}

// w = A * v, where A is the scaled 2D Laplacian (post-multiplied by 1/h²).
static void av(int nx, const double* v, double* w) {
    const int  n2 = nx * nx;
    const double inv_h2 = static_cast<double>((nx + 1) * (nx + 1));

    tv(nx, v, w);
    for (int k = 0; k < nx; ++k) w[k] -= v[nx + k];

    for (int j = 1; j < nx - 1; ++j) {
        const int lo = j * nx;
        tv(nx, v + lo, w + lo);
        for (int k = 0; k < nx; ++k) w[lo + k] -= v[lo - nx + k];
        for (int k = 0; k < nx; ++k) w[lo + k] -= v[lo + nx + k];
    }

    const int lo = (nx - 1) * nx;
    tv(nx, v + lo, w + lo);
    for (int k = 0; k < nx; ++k) w[lo + k] -= v[lo - nx + k];

    for (int k = 0; k < n2; ++k) w[k] *= inv_h2;
}

int main() {
    const int nx  = 10;
    const int n   = nx * nx;
    const int nev = 4;
    const int ncv = 10;

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> solver("I", n, "SM", nev, ncv);
    solver.tol(0.0).maxiter(300).mode(1).ishift(1);

    solver.solve([&](const double* x, double* y) { av(nx, x, y); });

    if (solver.info() < 0) {
        std::printf(" Error with _saupd, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs(true, 0.0);
    const int nconv = solver.num_converged();

    std::vector<double> ax(n);
    std::printf("\n Ritz values and relative residuals\n");
    std::printf(" ----------------------------------\n");
    for (int j = 0; j < nconv; ++j) {
        const double* xj = &r.vectors[j * n];
        av(nx, xj, ax.data());
        double nrm = 0.0;
        for (int k = 0; k < n; ++k) {
            double t = ax[k] - r.values[j] * xj[k];
            nrm += t * t;
        }
        nrm = std::sqrt(nrm) / std::abs(r.values[j]);
        std::printf(" Row%4d: %19.11e %19.11e\n", j + 1, r.values[j], nrm);
    }

    std::printf("\n _SDRV1 \n ====== \n");
    std::printf(" Size of the matrix is                  %d\n", n);
    std::printf(" The number of Ritz values requested    %d\n", nev);
    std::printf(" The number of Arnoldi vectors (NCV)    %d\n", ncv);
    std::printf(" What portion of the spectrum:          SM\n");
    std::printf(" The number of converged Ritz values    %d\n", nconv);
    std::printf(" Number of IA update iterations         %d\n", solver.num_iterations());
    std::printf(" The number of OP*x                     %d\n", solver.num_op_applies());
    return 0;
}
