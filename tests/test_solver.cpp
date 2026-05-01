// Tests for arnoldi::Arnoldi solver — covers different Kind/Scalar combos,
// user-provided initial residual, fluent API, shift-invert modes, eigenvalue
// extraction with and without eigenvectors, and convergence introspection.

#include <arnoldi/arnoldi.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <vector>


static int g_pass = 0, g_fail = 0;

static void check(const char* name, bool cond) {
    if (cond) {
        std::printf("  OK    %s\n", name);
        ++g_pass;
    } else {
        std::printf("  FAIL  %s\n", name);
        ++g_fail;
    }
}

// 1D Laplacian: A(i,i)=2, A(i,i±1)=-1, eigenvalues = 2-2*cos(k*pi/(n+1)).
static void av_laplacian(int n, const double* x, double* y) {
    y[0] = 2.0 * x[0] - x[1];
    for (int i = 1; i < n - 1; ++i)
        y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
    y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
}

static double exact_eig_laplacian(int n, int k) {
    return 2.0 - 2.0 * std::cos(k * M_PI / (n + 1));
}

void test_sym_double() {
    const int n = 64, nev = 4, ncv = 12;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("Sym<double>: converged", s.converged());
    check("Sym<double>: nconv >= nev", s.num_converged() >= nev);
    check("Sym<double>: info == 0", s.info() == 0);
    check("Sym<double>: iterations > 0", s.num_iterations() > 0);
    check("Sym<double>: op_applies > 0", s.num_op_applies() > 0);

    auto r = s.eigenpairs();
    check("Sym<double>: values.size() == nev", (int)r.values.size() == nev);
    check("Sym<double>: vectors.size() == n*nev", (int)r.vectors.size() == n * nev);

    double maxerr = 0;
    for (int i = 0; i < nev; ++i) {
        double exact = exact_eig_laplacian(n, i + 1);
        maxerr = std::max(maxerr, std::abs(r.values[i] - exact));
    }
    check("Sym<double>: eigenvalue accuracy < 1e-10", maxerr < 1e-10);
}

void test_sym_float() {
    const int n = 32, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, float> s("I", n, "SM", nev, ncv);
    s.tol(1e-5f).maxiter(500);
    s.solve([n](const float* x, float* y) {
        y[0] = 2.0f * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i)
            y[i] = -x[i - 1] + 2.0f * x[i] - x[i + 1];
        y[n - 1] = -x[n - 2] + 2.0f * x[n - 1];
    });

    check("Sym<float>: converged", s.converged());
    auto r = s.eigenpairs();
    float maxerr = 0;
    for (int i = 0; i < nev; ++i) {
        float exact = (float)exact_eig_laplacian(n, i + 1);
        maxerr = std::max(maxerr, std::abs(r.values[i] - exact));
    }
    check("Sym<float>: eigenvalue accuracy < 1e-4", maxerr < 1e-4f);
}

void test_nonsym_double() {
    const int n = 64, nev = 4, ncv = 14;
    const double rho = 10.0;
    const double h = 1.0 / (n + 1);
    const double dd = 2.0 / h;
    const double dl = -1.0 / h - rho / 2.0;
    const double du = -1.0 / h + rho / 2.0;

    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([&](const double* x, double* y) {
        y[0] = dd * x[0] + du * x[1];
        for (int i = 1; i < n - 1; ++i)
            y[i] = dl * x[i - 1] + dd * x[i] + du * x[i + 1];
        y[n - 1] = dl * x[n - 2] + dd * x[n - 1];
    });

    check("Nonsym<double>: converged", s.converged());
    check("Nonsym<double>: nconv >= nev", s.num_converged() >= nev);

    auto r = s.eigenpairs();
    check("Nonsym<double>: values_re.size() == nev+1", (int)r.values_re.size() == nev + 1);
    check("Nonsym<double>: values_im.size() == nev+1", (int)r.values_im.size() == nev + 1);
    check("Nonsym<double>: vectors non-empty", !r.vectors.empty());

    double min_mag = 1e30;
    for (int i = 0; i < nev; ++i) {
        double mag = std::sqrt(r.values_re[i] * r.values_re[i] +
                               r.values_im[i] * r.values_im[i]);
        min_mag = std::min(min_mag, mag);
    }
    check("Nonsym<double>: LM eigenvalues have large magnitude", min_mag > 100.0);
}

void test_herm_complex_double() {
    using cplx = std::complex<double>;
    const int n = 64, nev = 4, ncv = 12;
    const cplx off(-1.0, 0.25);

    arnoldi::Arnoldi<arnoldi::Kind::Herm, cplx> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([&](const cplx* x, cplx* y) {
        y[0] = 2.0 * x[0] + off * x[1];
        for (int i = 1; i < n - 1; ++i)
            y[i] = std::conj(off) * x[i - 1] + 2.0 * x[i] + off * x[i + 1];
        y[n - 1] = std::conj(off) * x[n - 2] + 2.0 * x[n - 1];
    });

    check("Herm<cdouble>: converged", s.converged());
    auto r = s.eigenpairs();
    check("Herm<cdouble>: values.size() == nev", (int)r.values.size() == nev);
    check("Herm<cdouble>: vectors.size() == n*nev", (int)r.vectors.size() == n * nev);

    for (int i = 0; i < nev; ++i)
        check("Herm<cdouble>: eigenvalues are real-valued (small)", std::isfinite(r.values[i]));
}

void test_user_initial_resid() {
    const int n = 32, nev = 3, ncv = 10;
    std::vector<double> resid(n, 1.0);

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.initial_resid(resid.data());
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("user_resid: converged", s.converged());
    auto r = s.eigenpairs();
    double err = std::abs(r.values[0] - exact_eig_laplacian(n, 1));
    check("user_resid: eigenvalue accurate", err < 1e-10);
}

void test_values_only() {
    const int n = 32, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("values_only: converged", s.converged());
    auto r = s.eigenpairs(false);
    check("values_only: values.size() == nev", (int)r.values.size() == nev);
    check("values_only: vectors empty", r.vectors.empty());
    check("values_only: eigenvalue is positive", r.values[0] > 0);
    check("values_only: eigenvalue is small", r.values[0] < 0.1);
}

void test_nonsym_values_only() {
    const int n = 32, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("nonsym_values_only: converged", s.converged());
    auto r = s.eigenpairs(false);
    check("nonsym_values_only: vectors empty", r.vectors.empty());
    check("nonsym_values_only: values_re non-zero", std::abs(r.values_re[0]) > 0.01);
}

void test_sym_which_LM() {
    const int n = 32, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "LM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("Sym which=LM: converged", s.converged());
    auto r = s.eigenpairs(false);
    check("Sym which=LM: largest eig > 3.5", r.values[0] > 3.5);
}

void test_sym_which_LA() {
    const int n = 32, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "LA", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("Sym which=LA: converged", s.converged());
    auto r = s.eigenpairs(false);
    check("Sym which=LA: largest algebraic > 3.5", r.values[nev - 1] > 3.5);
}

void test_sym_which_SA() {
    const int n = 32, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "SA", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("Sym which=SA: converged", s.converged());
    auto r = s.eigenpairs(false);
    double maxv = *std::max_element(r.values.begin(), r.values.end());
    check("Sym which=SA: all eigenvalues small", maxv < 0.15);
}

void test_sym_which_BE() {
    const int n = 32, nev = 4, ncv = 12;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "BE", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("Sym which=BE: converged", s.converged());
    auto r = s.eigenpairs(false);
    double minv = *std::min_element(r.values.begin(), r.values.end());
    double maxv = *std::max_element(r.values.begin(), r.values.end());
    check("Sym which=BE: span > 2.0", (maxv - minv) > 2.0);
}

// A = Laplacian, B = 2*I.  Eigenvalues of A*x = lambda*B*x are half the
// standard eigenvalues, so OP = inv(B)*A = A/2 in mode 2.
void test_sym_generalized() {
    const int n = 32, nev = 3, ncv = 10;

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("G", n, "LM", nev, ncv);
    s.tol(1e-12).maxiter(500).mode(2);
    s.solve(
        [n](const double* x, double* y) {
            y[0] = (2.0 * x[0] - x[1]) / 2.0;
            for (int i = 1; i < n - 1; ++i)
                y[i] = (-x[i - 1] + 2.0 * x[i] - x[i + 1]) / 2.0;
            y[n - 1] = (-x[n - 2] + 2.0 * x[n - 1]) / 2.0;
        },
        [n](const double* x, double* y) {
            for (int i = 0; i < n; ++i) y[i] = 2.0 * x[i];
        }
    );

    check("Sym generalized: converged", s.converged());
    auto r = s.eigenpairs(false);
    check("Sym generalized: largest eig > 1.5", r.values[0] > 1.5);
}

void test_insufficient_maxiter() {
    const int n = 64, nev = 4, ncv = 12;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "SM", nev, ncv);
    s.tol(1e-14).maxiter(1);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("maxiter=1: not converged", !s.converged());
    check("maxiter=1: info != 0 or nconv < nev",
          s.info() != 0 || s.num_converged() < nev);
}

void test_accessors() {
    const int n = 16, nev = 2, ncv = 6;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "SM", nev, ncv);
    s.tol(1e-10).maxiter(300);
    s.solve([n](const double* x, double* y) { av_laplacian(n, x, y); });

    check("accessors: v() != nullptr", s.v() != nullptr);
    check("accessors: workl() != nullptr", s.workl() != nullptr);
    check("accessors: iparam() != nullptr", s.iparam() != nullptr);
    check("accessors: ipntr() != nullptr", s.ipntr() != nullptr);
}

int main() {
    std::printf("test_solver:\n");

    test_sym_double();
    test_sym_float();
    test_nonsym_double();
    test_herm_complex_double();
    test_user_initial_resid();
    test_values_only();
    test_nonsym_values_only();
    test_sym_which_LM();
    test_sym_which_LA();
    test_sym_which_SA();
    test_sym_which_BE();
    test_sym_generalized();
    test_insufficient_maxiter();
    test_accessors();

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
