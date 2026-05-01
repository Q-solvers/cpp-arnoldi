// arnoldi_herm — smallest eigenvalues of a complex Hermitian tridiagonal.
//
//   H(i, i)   = 2
//   H(i, i+1) = -1 + i*alpha    (and its conjugate on the subdiagonal)
//
// The eigenvalues are real because H is Hermitian.
// Solved via arnoldi::Arnoldi<Kind::Herm, std::complex<double>>.

#include <arnoldi/arnoldi.hpp>

#include <complex>
#include <cstdio>

int main() {
    using cplx = std::complex<double>;

    const int n   = 128;
    const int nev = 5;
    const int ncv = 2 * nev + 2;

    const cplx off(-1.0, 0.25);

    arnoldi::Arnoldi<arnoldi::Kind::Herm, cplx> solver("I", n, "SM", nev, ncv);
    solver.tol(0.0).maxiter(500);

    solver.solve([&](const cplx* x, cplx* y) {
        y[0] = 2.0 * x[0] + off * x[1];
        for (int i = 1; i < n - 1; ++i)
            y[i] = std::conj(off) * x[i - 1] + 2.0 * x[i] + off * x[i + 1];
        y[n - 1] = std::conj(off) * x[n - 2] + 2.0 * x[n - 1];
    });

    if (!solver.converged()) {
        std::printf("saupd<complex> failed, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs();
    std::printf("Smallest %d eigenvalues (complex Hermitian tridiag, n=%d):\n", nev, n);
    for (int i = 0; i < nev; ++i)
        std::printf("  lambda[%d] = %.12f\n", i, r.values[i]);
    std::printf("iterations=%d, OP applies=%d\n",
                solver.num_iterations(), solver.num_op_applies());
    return 0;
}
