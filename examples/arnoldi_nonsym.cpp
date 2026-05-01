// arnoldi_nonsym — largest-magnitude eigenvalues of the nonsymmetric
// convection-diffusion operator on a 1D grid.
//
//   A  =  tridiag(-1-rho*h/2,  2,  -1+rho*h/2)  (scaled by 1/h^2)
//
// via arnoldi::Arnoldi<Kind::Nonsym, double>.

#include <arnoldi/arnoldi.hpp>

#include <cstdio>

int main() {
    const int n   = 128;
    const int nev = 4;
    const int ncv = 20;

    const double rho = 10.0;
    const double h   = 1.0 / (n + 1);
    const double h2  = h * h;
    const double sub = -1.0 / h2 - rho / (2.0 * h);
    const double sup = -1.0 / h2 + rho / (2.0 * h);
    const double dia =  2.0 / h2;

    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> solver("I", n, "LM", nev, ncv);
    solver.tol(0.0).maxiter(300);

    solver.solve([&](const double* x, double* y) {
        y[0] = dia * x[0] + sup * x[1];
        for (int i = 1; i < n - 1; ++i)
            y[i] = sub * x[i - 1] + dia * x[i] + sup * x[i + 1];
        y[n - 1] = sub * x[n - 2] + dia * x[n - 1];
    });

    if (!solver.converged()) {
        std::printf("naupd failed, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs();
    std::printf("Largest %d eigenvalues (convection-diffusion, n=%d, rho=%.1f):\n",
                nev, n, rho);
    for (int i = 0; i < nev; ++i)
        std::printf("  lambda[%d] = %.12g %+.12g i\n", i, r.values_re[i], r.values_im[i]);
    std::printf("iterations=%d, OP applies=%d\n",
                solver.num_iterations(), solver.num_op_applies());
    return 0;
}
