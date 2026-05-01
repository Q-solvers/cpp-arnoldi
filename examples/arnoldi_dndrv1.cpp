// C++ port of EXAMPLES/NONSYM/dndrv1.f using arnoldi::Arnoldi<Kind::Nonsym, double>.
//
// Standard nonsymmetric eigenvalue problem  A*x = lambda*x  (regular mode, Mode 1).
// A is the 2D convection-diffusion operator on the unit square with Dirichlet
// boundary conditions, on an nx × nx interior grid:
//
//    A*u  =  -Δu + rho * du/dx     (rho = 0 here → reduces to real Laplacian)
//
// Discretisation uses 5-point central differences.  When rho*h/2 > 1 the
// eigenvalues become genuinely complex; rho=0 keeps them all real.

#include <arnoldi/arnoldi.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

static void tv(int nx, double dd, double dl, double du,
               const double* x, double* y) {
    y[0] = dd * x[0] + du * x[1];
    for (int j = 1; j < nx - 1; ++j)
        y[j] = dl * x[j - 1] + dd * x[j] + du * x[j + 1];
    y[nx - 1] = dl * x[nx - 2] + dd * x[nx - 1];
}

static void av(int nx, const double* v, double* w) {
    constexpr double rho = 0.0;
    const double h    = 1.0 / (nx + 1);
    const double h2   = h * h;
    const double dd   = 4.0 / h2;
    const double dl   = -1.0 / h2 - 0.5 * rho / h;
    const double du   = -1.0 / h2 + 0.5 * rho / h;
    const double coup = -1.0 / h2;

    tv(nx, dd, dl, du, v, w);
    for (int k = 0; k < nx; ++k) w[k] += coup * v[nx + k];

    for (int j = 1; j < nx - 1; ++j) {
        const int lo = j * nx;
        tv(nx, dd, dl, du, v + lo, w + lo);
        for (int k = 0; k < nx; ++k) w[lo + k] += coup * v[lo - nx + k];
        for (int k = 0; k < nx; ++k) w[lo + k] += coup * v[lo + nx + k];
    }

    const int lo = (nx - 1) * nx;
    tv(nx, dd, dl, du, v + lo, w + lo);
    for (int k = 0; k < nx; ++k) w[lo + k] += coup * v[lo - nx + k];
}

int main() {
    const int nx  = 10;
    const int n   = nx * nx;
    const int nev = 4;
    const int ncv = 20;

    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> solver("I", n, "SM", nev, ncv);
    solver.tol(0.0).maxiter(300).mode(1).ishift(1);

    solver.solve([&](const double* x, double* y) { av(nx, x, y); });

    if (solver.info() < 0) {
        std::printf(" Error with _naupd, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs(true, 0.0, 0.0);
    const int nconv = solver.num_converged();

    // Relative residuals, handling conjugate pairs like dneupd does.
    std::vector<double> ax(n);
    std::printf("\n Ritz values (Real, Imag) and relative residuals\n");
    std::printf(" -----------------------------------------------\n");
    for (int j = 0; j < nconv; ) {
        const double* xj = &r.vectors[j * n];
        const double lr  = r.values_re[j];
        const double li  = r.values_im[j];

        if (li == 0.0) {
            av(nx, xj, ax.data());
            double s = 0.0;
            for (int k = 0; k < n; ++k) {
                double t = ax[k] - lr * xj[k];
                s += t * t;
            }
            const double nrm = std::sqrt(s) / std::abs(lr);
            std::printf(" Row%4d: %19.11e %19.11e %19.11e\n",
                        j + 1, lr, li, nrm);
            ++j;
        } else {
            // Conjugate pair: eigenvector = xj + i * x_{j+1}.
            // Compute A*(xj + i*xi) and subtract (lr + i*li)*(xj + i*xi).
            const double* xi = &r.vectors[(j + 1) * n];
            std::vector<double> axi(n);
            av(nx, xj, ax.data());
            av(nx, xi, axi.data());
            double s = 0.0;
            for (int k = 0; k < n; ++k) {
                double re = ax[k]  - (lr * xj[k] - li * xi[k]);
                double im = axi[k] - (li * xj[k] + lr * xi[k]);
                s += re * re + im * im;
            }
            const double nrm = std::sqrt(s) / std::hypot(lr, li);
            std::printf(" Row%4d: %19.11e %19.11e %19.11e\n",
                        j + 1, lr,  li, nrm);
            std::printf(" Row%4d: %19.11e %19.11e %19.11e\n",
                        j + 2, lr, -li, nrm);
            j += 2;
        }
    }

    std::printf("\n _NDRV1 \n ====== \n");
    std::printf(" Size of the matrix is                  %d\n", n);
    std::printf(" The number of Ritz values requested    %d\n", nev);
    std::printf(" The number of Arnoldi vectors (NCV)    %d\n", ncv);
    std::printf(" What portion of the spectrum:          SM\n");
    std::printf(" The number of converged Ritz values    %d\n", nconv);
    std::printf(" Number of IA update iterations         %d\n", solver.num_iterations());
    std::printf(" The number of OP*x                     %d\n", solver.num_op_applies());
    return 0;
}
