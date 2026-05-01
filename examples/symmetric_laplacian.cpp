// Smallest-magnitude eigenvalues of the 1D Dirichlet Laplacian (-1, 2, -1).
#include <arnoldi/detail.hpp>

#include <cstdio>
#include <vector>

static void laplacian_matvec(int n, const double* x, double* y) {
    y[0] = 2.0 * x[0] - x[1];
    for (int i = 1; i < n - 1; i++)
        y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
    y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
}

int main() {
    const int n = 128;
    const int nev = 5;
    const int ncv = std::min(2 * nev + 1, n);
    const int lworkl = ncv * ncv + 8 * ncv;

    std::vector<double> resid(n, 0.0);
    std::vector<double> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(ncv), 0.0);
    std::vector<double> workd(3 * n, 0.0);
    std::vector<double> workl(lworkl, 0.0);
    std::vector<double> d(nev, 0.0);
    std::vector<double> z(static_cast<std::size_t>(n) * static_cast<std::size_t>(nev), 0.0);
    int iparam[11] = {};
    int ipntr[11]  = {};

    iparam[0] = 1;
    iparam[2] = 300;
    iparam[6] = 1;

    int info = 0;
    double tol = 0.0; // saupd takes tol by reference, so it cannot be const

    auto op = [&](const double* x, double* y) { laplacian_matvec(n, x, y); };

    arnoldi::detail::saupd<double>("I", n, "SM", nev, tol, resid.data(), ncv,
                              v.data(), n, iparam, ipntr, workd.data(),
                              workl.data(), lworkl, info, op);

    if (info < 0) {
        std::printf("saupd failed, info = %d\n", info);
        return 1;
    }

    const double sigma = 0.0;
    arnoldi::detail::seupd<double>(true, "A", d.data(), z.data(), n, sigma,
                              "I", n, "SM", nev, tol, resid.data(), ncv, v.data(), n,
                              iparam, ipntr, workd.data(), workl.data(), lworkl, info);

    if (info != 0) {
        std::printf("seupd failed, info = %d\n", info);
        return 1;
    }

    std::printf("Smallest %d eigenvalues (symmetric, SM):\n", nev);
    for (int i = 0; i < nev; i++)
        std::printf("  lambda[%d] = %.12f\n", i, d[i]);

    return 0;
}
