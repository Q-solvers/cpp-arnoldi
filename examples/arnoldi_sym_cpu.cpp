// arnoldi_sym_cpu — CPU twin of examples/arnoldi_sym_cuda.cu.
//
// Same problem (diagonal operator A(i,i) = i+1, n = 8192, nev = 5,
// ncv = 8*nev, "LM", tol 1e-10) solved with the default CpuBackend and a
// host matvec. Times the solve() call so it can be compared directly
// against the CUDA example as a backend performance benchmark.
//
// Build: configure with -DARNOLDI_BUILD_EXAMPLES=ON (no CUDA needed).

#include <arnoldi/arnoldi.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

int main() {
    const int n   = 1 << 18;  // 8192 — same size as the CUDA example
    const int nev = 5;
    const int ncv = 8 * nev;  // matches arnoldi_sym_cuda.cu

    // "LM": the nev largest eigenvalues, i.e. n, n-1, ..., n-nev+1.
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> solver("I", n, "LM", nev, ncv);
    solver.tol(1e-10).maxiter(1000);

    auto t0 = std::chrono::steady_clock::now();
    solver.solve([n](const double* x, double* y) {
        for (int i = 0; i < n; ++i) y[i] = static_cast<double>(i + 1) * x[i];
    });
    auto t1     = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    if (!solver.converged()) {
        std::printf("CPU saupd failed, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs(false);  // values only
    std::sort(r.values.begin(), r.values.end(), std::greater<double>());
    std::printf("Largest %d eigenvalues of diag(1..%d), CPU backend:\n", nev, n);
    for (int i = 0; i < nev; ++i) {
        double exact = static_cast<double>(n - i);
        std::printf("  lambda[%d] = %.6f   (exact %.0f, |err| = %.2e)\n",
                    i, r.values[i], exact, std::fabs(r.values[i] - exact));
    }
    std::printf("iterations=%d, OP applies=%d\n",
                solver.num_iterations(), solver.num_op_applies());
    std::printf("solve() wall time: %.4f s\n", secs);
    return 0;
}
