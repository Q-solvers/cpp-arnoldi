// Unit test for getv0 — exercises restart, orthogonalization, generalized,
// and debug paths that are not reached by the standard examples.

#include <arnoldi/detail/getv0.hpp>
#include <arnoldi/comm.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using arnoldi::detail::Ops;

// Identity operator for the standard eigenvalue problem.
static void op_identity(const double* x, double* y) {
    const int n = 8;
    for (int i = 0; i < n; ++i) y[i] = x[i];
}

// B = diag(1..n).
static void bop_diag(const double* x, double* y) {
    const int n = 8;
    for (int i = 0; i < n; ++i) y[i] = (i + 1) * x[i];
}

static bool test_basic() {
    const int n = 8;
    std::vector<double> resid(n), workd(2 * n), v(n);
    double rnorm = 0;
    int ierr = 0;
    arnoldi::SerialComm comm;

    arnoldi::detail::getv0<double>("I", 1, false, n, 1,
        v.data(), n, resid.data(), rnorm, workd.data(), ierr,
        op_identity, [](const double*, double*){}, comm);

    assert(ierr == 0);
    assert(rnorm > 0);
    std::printf("  basic:         rnorm=%.6e  OK\n", rnorm);
    return true;
}

// itry > 1 with bmat='G' takes the else-if branch in getv0.
static bool test_generalized_retry() {
    const int n = 8;
    std::vector<double> resid(n), workd(2 * n), v(n);
    double rnorm = 0;
    int ierr = 0;
    arnoldi::SerialComm comm;

    for (int i = 0; i < n; ++i) resid[i] = 1.0;
    arnoldi::detail::getv0<double>("G", 2, true, n, 1,
        v.data(), n, resid.data(), rnorm, workd.data(), ierr,
        op_identity, bop_diag, comm);

    assert(ierr == 0);
    assert(rnorm > 0);
    std::printf("  gen retry:     rnorm=%.6e  OK\n", rnorm);
    return true;
}

// j != 1 takes the orthogonalization-against-existing-basis path.
static bool test_orthogonalization() {
    const int n = 8;
    const int j = 2;
    std::vector<double> v(n * j, 0.0);
    std::vector<double> resid(n), workd(2 * n);
    double rnorm = 0;
    int ierr = 0;
    arnoldi::SerialComm comm;

    v[0] = 1.0;
    for (int i = 0; i < n; ++i) resid[i] = 1.0;

    arnoldi::detail::getv0<double>("I", 1, true, n, j,
        v.data(), n, resid.data(), rnorm, workd.data(), ierr,
        op_identity, [](const double*, double*){}, comm);

    assert(ierr == 0);
    assert(rnorm > 0);

    double dot = Ops<double>::dot(n, resid.data(), 1, v.data(), 1);
    assert(std::abs(dot) < 1e-12);
    std::printf("  orthog:        rnorm=%.6e  dot=%.2e  OK\n", rnorm, dot);
    return true;
}

// Same orthogonalization path with bmat='G'.
static bool test_orthogonalization_generalized() {
    const int n = 8;
    const int j = 2;
    std::vector<double> v(n * j, 0.0);
    std::vector<double> resid(n), workd(2 * n);
    double rnorm = 0;
    int ierr = 0;
    arnoldi::SerialComm comm;

    v[0] = 1.0;
    for (int i = 0; i < n; ++i) resid[i] = 1.0;

    arnoldi::detail::getv0<double>("G", 1, true, n, j,
        v.data(), n, resid.data(), rnorm, workd.data(), ierr,
        op_identity, bop_diag, comm);

    assert(ierr == 0);
    assert(rnorm > 0);
    std::printf("  orthog gen:    rnorm=%.6e  OK\n", rnorm);
    return true;
}

// msglvl > 3 triggers both the rnorm and full-vector debug prints.
static bool test_debug_output() {
    const int n = 8;
    std::vector<double> resid(n), workd(2 * n), v(n);
    double rnorm = 0;
    int ierr = 0;
    arnoldi::SerialComm comm;

    int saved = arnoldi::detail::debug.getv0;
    arnoldi::detail::debug.getv0 = 4;

    arnoldi::detail::getv0<double>("I", 1, false, n, 1,
        v.data(), n, resid.data(), rnorm, workd.data(), ierr,
        op_identity, [](const double*, double*){}, comm);

    arnoldi::detail::debug.getv0 = saved;

    assert(ierr == 0);
    assert(rnorm > 0);
    std::printf("  debug:         rnorm=%.6e  OK\n", rnorm);
    return true;
}

int main() {
    std::printf("test_getv0:\n");
    bool ok = true;
    ok &= test_basic();
    ok &= test_generalized_retry();
    ok &= test_orthogonalization();
    ok &= test_orthogonalization_generalized();
    ok &= test_debug_output();
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
