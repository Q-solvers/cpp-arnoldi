// Tests for the shift-invert, buckling, and Cayley extraction paths in
// arnoldi::Arnoldi. The default test_solver only drives Mode 1 (regular)
// for both Sym and Nonsym; this file walks Modes 2-5 plus a few rvec=false
// variants so seupd/neupd's transform branches are exercised.

#include <arnoldi/arnoldi.hpp>
#include <arnoldi/detail/debug.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

extern "C" {
void dgttrf_(const int* n, double* dl, double* d, double* du, double* du2, int* ipiv, int* info);
void dgttrs_(const char* trans, const int* n, const int* nrhs, const double* dl, const double* d, const double* du,
             const double* du2, const int* ipiv, double* b, const int* ldb, int* info);
}

#include <catch2/catch_test_macros.hpp>

// Bridge legacy check("desc", cond) calls onto Catch2 assertions.
#define check(msg, cond) \
    do {                 \
        INFO(msg);       \
        CHECK((cond));   \
    } while (0)

// ---------------------------------------------------------------------------
// Analytic spectra of the test problems.
// ---------------------------------------------------------------------------

// Eigenvalues of the n×n tridiagonal matrix [-1, 2, -1] (Dirichlet).
//   λ_k = 4 sin²(kπ / 2(n+1)),  k = 1..n.
static std::vector<double> tridiag_lap_spectrum(int n) {
    std::vector<double> e(n);
    for (int k = 1; k <= n; ++k) {
        double s = std::sin(k * M_PI / (2.0 * (n + 1)));
        e[k - 1] = 4.0 * s * s;
    }
    return e;
}

// Eigenvalues of the generalized FE Laplacian Ax = λMx with Dirichlet BCs on
// [0,1], uniform mesh, h = 1/(n+1):
//   A = (1/h) tridiag(-1, 2, -1),   M = (h/6) tridiag(1, 4, 1)
//   λ_k = (6/h²) (1 − cos(kπh)) / (2 + cos(kπh)),  k = 1..n.
static std::vector<double> fe_lap_spectrum(int n) {
    std::vector<double> e(n);
    const double h = 1.0 / (n + 1);
    for (int k = 1; k <= n; ++k) {
        double c = std::cos(k * M_PI * h);
        e[k - 1] = (6.0 / (h * h)) * (1.0 - c) / (2.0 + c);
    }
    return e;
}

// Imaginary parts of eigenvalues of the n×n skew-symmetric tridiagonal matrix
// with super-diagonal +1 and sub-diagonal −1 (zero diagonal). Real parts are
// zero and the imaginary parts are 2 cos(kπ/(n+1)) for k = 1..n (signed,
// so they appear as conjugate pairs).
static std::vector<double> skew_tridiag_imag_spectrum(int n) {
    std::vector<double> e(n);
    for (int k = 1; k <= n; ++k) e[k - 1] = 2.0 * std::cos(k * M_PI / (n + 1));
    return e;
}

// Pick the nev values from `all` that ARPACK would converge to given a
// selection rule. For real spectra:
//   "LM" / "SM" / "LA" / "SA" / "BE" follow ARPACK semantics.
//   "NEAR" picks the nev values with smallest |λ - sigma| (shift-invert
//          and buckling: μ = 1/(λ-σ) or λ/(λ-σ) - largest |μ| is closest λ).
//   "CAYLEY" picks by largest |μ| with μ = (λ + σ) / (λ − σ).
static std::vector<double> pick_real(std::vector<double> all, int nev,
                                     const char* mode, double sigma = 0.0) {
    std::vector<double> sorted = all;
    auto byabs_desc = [](double a, double b) { return std::abs(a) > std::abs(b); };
    auto byabs_asc  = [](double a, double b) { return std::abs(a) < std::abs(b); };
    if (std::string(mode) == "LM") std::sort(sorted.begin(), sorted.end(), byabs_desc);
    else if (std::string(mode) == "SM") std::sort(sorted.begin(), sorted.end(), byabs_asc);
    else if (std::string(mode) == "LA") std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    else if (std::string(mode) == "SA") std::sort(sorted.begin(), sorted.end());
    else if (std::string(mode) == "NEAR") {
        std::sort(sorted.begin(), sorted.end(),
                  [sigma](double a, double b) { return std::abs(a - sigma) < std::abs(b - sigma); });
    } else if (std::string(mode) == "CAYLEY") {
        auto mu = [sigma](double l) {
            double d = l - sigma;
            return (d == 0.0) ? std::numeric_limits<double>::infinity() : (l + sigma) / d;
        };
        std::sort(sorted.begin(), sorted.end(),
                  [&](double a, double b) { return std::abs(mu(a)) > std::abs(mu(b)); });
    } else if (std::string(mode) == "BE") {
        std::sort(sorted.begin(), sorted.end());
        std::vector<double> out;
        const int half_lo = nev / 2;
        const int half_hi = nev - half_lo;
        for (int i = 0; i < half_lo; ++i) out.push_back(sorted[i]);
        for (int i = 0; i < half_hi; ++i) out.push_back(sorted[(int)sorted.size() - 1 - i]);
        return out;
    }
    sorted.resize(nev);
    return sorted;
}

// Compare two real eigenvalue sets of the same size using a sorted L_inf
// metric scaled by the spectral magnitude. Returns max relative error.
static double compare_real(std::vector<double> got, std::vector<double> want) {
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    double scale = 0.0, err = 0.0;
    for (double v : want) scale = std::max(scale, std::abs(v));
    if (scale == 0.0) scale = 1.0;
    const int m = (int)std::min(got.size(), want.size());
    for (int i = 0; i < m; ++i) err = std::max(err, std::abs(got[i] - want[i]) / scale);
    return err;
}

// Compare complex eigenvalue magnitudes (for conjugate-pair spectra).
static double compare_abs(std::vector<double> got, std::vector<double> want) {
    for (auto& v : got) v = std::abs(v);
    for (auto& v : want) v = std::abs(v);
    return compare_real(std::move(got), std::move(want));
}

// Pre-factored (A - sigma*M) tridiagonal solver: A is the FE Laplacian and
// M is the FE mass matrix on a uniform 1-D grid with Dirichlet BCs.
struct ShiftInvertFE {
    int n;
    double sigma;
    std::vector<double> d, dl, du, du2;
    std::vector<int>    ipiv;

    ShiftInvertFE(int n_, double sigma_) : n(n_), sigma(sigma_), d(n), dl(n), du(n), du2(n, 0.0), ipiv(n) {
        const double h  = 1.0 / (n + 1);
        const double r1 = (4.0 / 6.0) * h;
        const double r2 = (1.0 / 6.0) * h;
        for (int j = 0; j < n; ++j) {
            d[j]  = 2.0 / h - sigma * r1;
            dl[j] = -1.0 / h - sigma * r2;
        }
        std::copy(dl.begin(), dl.end(), du.begin());
        int info = 0;
        dgttrf_(&n, dl.data(), d.data(), du.data(), du2.data(), ipiv.data(), &info);
        if (info != 0) std::printf("ShiftInvertFE: gttrf info=%d\n", info);
    }

    void solve(double* rhs) const {
        int info = 0, nrhs = 1, ldb = n;
        dgttrs_("N", &n, &nrhs, dl.data(), d.data(), du.data(), du2.data(), ipiv.data(), rhs, &ldb, &info);
    }
};

static void av_lap(int n, const double* v, double* w) {
    const double h = 1.0 / (n + 1);
    w[0] = (2.0 / h) * v[0] - (1.0 / h) * v[1];
    for (int j = 1; j < n - 1; ++j)
        w[j] = -(1.0 / h) * v[j - 1] + (2.0 / h) * v[j] - (1.0 / h) * v[j + 1];
    w[n - 1] = -(1.0 / h) * v[n - 2] + (2.0 / h) * v[n - 1];
}

static void mv_mass(int n, const double* v, double* w) {
    const double h = 1.0 / (n + 1);
    w[0] = (4.0 * h / 6.0) * v[0] + (h / 6.0) * v[1];
    for (int j = 1; j < n - 1; ++j)
        w[j] = (h / 6.0) * v[j - 1] + (4.0 * h / 6.0) * v[j] + (h / 6.0) * v[j + 1];
    w[n - 1] = (h / 6.0) * v[n - 2] + (4.0 * h / 6.0) * v[n - 1];
}

// Mode 3: A*x = lambda*M*x via OP = inv(A - sigma*M)*M, B = M.
// Shift-invert with σ=0 converges to the eigenvalues of (A,M) closest to 0,
// i.e. the smallest eigenvalues of the FE Laplacian.
TEST_CASE("test_sym_mode3_shift_invert", "[modes]") {
    const int    n = 100, nev = 4, ncv = 10;
    const double sigma = 0.0;
    ShiftInvertFE shift(n, sigma);

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("G", n, "LM", nev, ncv);
    s.tol(0.0).maxiter(300).mode(3).ishift(1);

    auto op  = [&](const double* x, double* y) { mv_mass(n, x, y); shift.solve(y); };
    auto bop = [&](const double* x, double* y) { mv_mass(n, x, y); };

    s.solve(op, bop);
    check("Sym mode 3: solve ok", s.info() >= 0);

    auto r = s.eigenpairs(true, sigma);
    check("Sym mode 3: nconv >= nev", s.num_converged() >= nev);

    auto exact = pick_real(fe_lap_spectrum(n), nev, "NEAR", sigma);
    double err = compare_real(r.values, exact);
    std::printf("    sym mode 3 max rel err = %.3e\n", err);
    check("Sym mode 3: matches exact FE spectrum", err < 1e-9);

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s2("G", n, "LM", nev, ncv);
    s2.tol(0.0).maxiter(300).mode(3).ishift(1);
    s2.solve(op, bop);
    auto r2 = s2.eigenpairs(false, sigma);
    double err2 = compare_real(r2.values, exact);
    check("Sym mode 3 rvec=false: matches exact", r2.vectors.empty() && err2 < 1e-9);
}

// Mode 4 (buckling): K*x = lambda*KG*x via OP = inv(K - sigma*KG)*K, B = K.
// In our setup K = FE stiffness, KG = FE mass, so the underlying spectrum is
// still the FE Laplacian spectrum and the converged values are nearest σ.
TEST_CASE("test_sym_mode4_buckling", "[modes]") {
    const int    n = 100, nev = 4, ncv = 10;
    const double sigma = 1.0;
    ShiftInvertFE shift(n, sigma);

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("G", n, "LM", nev, ncv);
    s.tol(0.0).maxiter(300).mode(4).ishift(1);

    auto op  = [&](const double* x, double* y) { av_lap(n, x, y); shift.solve(y); };
    auto bop = [&](const double* x, double* y) { av_lap(n, x, y); };

    s.solve(op, bop);
    check("Sym mode 4: solve ok", s.info() >= 0);
    auto r = s.eigenpairs(true, sigma);

    auto exact = pick_real(fe_lap_spectrum(n), nev, "NEAR", sigma);
    double err = compare_real(r.values, exact);
    std::printf("    sym mode 4 max rel err = %.3e\n", err);
    check("Sym mode 4: matches exact FE spectrum", err < 1e-9);

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s2("G", n, "LM", nev, ncv);
    s2.tol(0.0).maxiter(300).mode(4).ishift(1);
    s2.solve(op, bop);
    auto r2 = s2.eigenpairs(false, sigma);
    double err2 = compare_real(r2.values, exact);
    check("Sym mode 4 rvec=false: matches exact", r2.vectors.empty() && err2 < 1e-9);
}

// Mode 5 (Cayley): A*x = lambda*M*x via OP = inv(A - sigma*M)*(A + sigma*M).
// μ = (λ+σ)/(λ−σ); LM Ritz values give λ closest to σ.
TEST_CASE("test_sym_mode5_cayley", "[modes]") {
    const int    n = 100, nev = 4, ncv = 20;
    const double sigma = 150.0;
    ShiftInvertFE shift(n, sigma);

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("G", n, "LM", nev, ncv);
    s.tol(0.0).maxiter(300).mode(5).ishift(1);

    auto op = [&](const double* x, double* y) {
        std::vector<double> tmp(n);
        av_lap(n, x, y);
        mv_mass(n, x, tmp.data());
        for (int k = 0; k < n; ++k) y[k] += sigma * tmp[k];
        shift.solve(y);
    };
    auto bop = [&](const double* x, double* y) { mv_mass(n, x, y); };

    s.solve(op, bop);
    check("Sym mode 5: solve ok", s.info() >= 0);
    auto r = s.eigenpairs(true, sigma);

    auto exact = pick_real(fe_lap_spectrum(n), nev, "CAYLEY", sigma);
    double err = compare_real(r.values, exact);
    std::printf("    sym mode 5 max rel err = %.3e\n", err);
    check("Sym mode 5: matches exact FE spectrum", err < 1e-8);
}

// Sym Mode 2 (generalized regular).  bmat='G' but no shift-invert; OP = M^-1*A.
// ARPACK mode-2 protocol: the OP callback must overwrite its input buffer
// with A*x because saitr later reads it back when computing the M-norm
// (wnorm² = (M^-1 A x)·x where x is required to be A*x at that point).
// A = standard tridiagonal Laplacian, M = 2*I, so eigenvalues are λ(A)/2.
TEST_CASE("test_sym_mode2_generalized", "[modes]") {
    const int n = 64, nev = 3, ncv = 10;

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("G", n, "LM", nev, ncv);
    s.tol(1e-10).maxiter(500).mode(2).ishift(1);

    auto op = [n](const double* x_in, double* y) {
        double* x = const_cast<double*>(x_in);
        std::vector<double> ax(n);
        ax[0] = 2.0 * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i) ax[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
        ax[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
        std::copy(ax.begin(), ax.end(), x);
        for (int i = 0; i < n; ++i) y[i] = ax[i] / 2.0;
    };
    auto bop = [n](const double* x, double* y) { for (int i = 0; i < n; ++i) y[i] = 2.0 * x[i]; };

    s.solve(op, bop);
    check("Sym mode 2: converged", s.converged());
    auto r = s.eigenpairs(true);

    auto full = tridiag_lap_spectrum(n);
    for (auto& v : full) v *= 0.5;
    auto exact = pick_real(full, nev, "LM");
    double err = compare_real(r.values, exact);
    std::printf("    sym mode 2 max rel err = %.3e\n", err);
    check("Sym mode 2: matches exact spectrum", err < 1e-9);
}

// Nonsymmetric shift-invert (mode 3, real shift).  The underlying problem is
// still symmetric (FE Laplacian generalized), so all eigenvalues are real and
// nev nearest σ should appear in r.values_re with values_im ≈ 0.
TEST_CASE("test_nonsym_mode3_real_shift", "[modes]") {
    const int    n = 100, nev = 4, ncv = 20;
    const double sigma = 1.0;
    ShiftInvertFE shift(n, sigma);

    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("G", n, "LM", nev, ncv);
    s.tol(0.0).maxiter(500).mode(3).ishift(1);

    auto op  = [&](const double* x, double* y) { mv_mass(n, x, y); shift.solve(y); };
    auto bop = [&](const double* x, double* y) { mv_mass(n, x, y); };

    s.solve(op, bop);
    check("Nonsym mode 3 real: solve ok", s.info() >= 0);

    auto r = s.eigenpairs(true, sigma, 0.0);
    check("Nonsym mode 3 real: nconv > 0", s.num_converged() > 0);

    auto exact = pick_real(fe_lap_spectrum(n), nev, "NEAR", sigma);
    std::vector<double> got(r.values_re.begin(), r.values_re.begin() + nev);
    double err = compare_real(got, exact);
    double max_im = 0.0;
    for (int i = 0; i < nev; ++i) max_im = std::max(max_im, std::abs(r.values_im[i]));
    std::printf("    nonsym mode 3 max rel err = %.3e, max |im| = %.3e\n", err, max_im);
    check("Nonsym mode 3: matches exact FE spectrum", err < 1e-9);
    check("Nonsym mode 3: imag parts ~ 0", max_im < 1e-9);

    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s2("G", n, "LM", nev, ncv);
    s2.tol(0.0).maxiter(500).mode(3).ishift(1);
    s2.solve(op, bop);
    auto r2 = s2.eigenpairs(false, sigma, 0.0);
    std::vector<double> got2(r2.values_re.begin(), r2.values_re.begin() + nev);
    check("Nonsym mode 3 rvec=false: matches exact",
          r2.vectors.empty() && compare_real(got2, exact) < 1e-9);
}

// Sym which="LM" with rvec=false hits the REGULR branch of seupd that copies
// ritz into d, distinct from the rvec=true sesrt path.
TEST_CASE("test_sym_regular_rvec_false_LM", "[modes]") {
    const int n = 32, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "LM", nev, ncv);
    s.tol(1e-10).maxiter(500);
    s.solve([n](const double* x, double* y) {
        y[0] = 2.0 * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
        y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
    });
    auto r = s.eigenpairs(false);

    auto exact = pick_real(tridiag_lap_spectrum(n), nev, "LM");
    double err = compare_real(r.values, exact);
    std::printf("    sym REGULR rvec=false LM max rel err = %.3e\n", err);
    check("Sym REGULR rvec=false LM: vectors empty", r.vectors.empty());
    check("Sym REGULR rvec=false LM: matches exact spectrum", err < 1e-10);
}

// Nonsym with which="SM" exercises the SM mapping in ngets and the regular
// extraction path.  Underlying matrix is symmetric so eigenvalues are real.
TEST_CASE("test_nonsym_which_SM", "[modes]") {
    const int n = 32, nev = 3, ncv = 12;
    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "SM", nev, ncv);
    s.tol(1e-10).maxiter(500);
    s.solve([n](const double* x, double* y) {
        y[0] = 2.0 * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
        y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
    });
    check("Nonsym SM: solve ok", s.info() >= 0 && s.num_converged() > 0);
    auto r = s.eigenpairs(true);

    auto exact = pick_real(tridiag_lap_spectrum(n), nev, "SM");
    std::vector<double> got(r.values_re.begin(), r.values_re.begin() + nev);
    double err = compare_real(got, exact);
    std::printf("    nonsym SM max rel err = %.3e\n", err);
    check("Nonsym SM: matches exact spectrum", err < 1e-9);
}

// Herm with which="LM" + rvec=false; covers the saupd<complex> regular path
// and the rvec=false copy-Ritz path in seupd<complex>.
TEST_CASE("test_herm_LM_LA", "[modes]") {
    using cplx = std::complex<double>;
    const int n = 32, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Herm, cplx> s("I", n, "LM", nev, ncv);
    s.tol(1e-10).maxiter(500);
    s.solve([n](const cplx* x, cplx* y) {
        y[0] = 2.0 * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
        y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
    });
    check("Herm LM: converged", s.converged());
    auto r = s.eigenpairs(false);

    auto exact = pick_real(tridiag_lap_spectrum(n), nev, "LM");
    double err = compare_real(r.values, exact);
    std::printf("    herm LM max rel err = %.3e\n", err);
    check("Herm LM: rvec=false vectors empty", r.vectors.empty());
    check("Herm LM: matches exact spectrum", err < 1e-10);
}

// Helper RAII guard that turns every debug.* msglvl up to MAX while in scope
// and restores the previous values on exit. Lets us exercise all the
// "if (msglvl > N) ..." debug-print branches inside *aitr/*aupd/*eupd.
struct DebugAllOn {
    int aupd, aup2, aitr, eigh, gets, apps, eupd, getv0_;
    DebugAllOn() {
        using namespace arnoldi::detail;
        aupd = debug.aupd; aup2 = debug.aup2; aitr = debug.aitr;
        eigh = debug.eigh; gets = debug.gets; apps = debug.apps;
        eupd = debug.eupd; getv0_ = debug.getv0;
        debug.aupd = debug.aup2 = debug.aitr = 4;
        debug.eigh = debug.gets = debug.apps = 4;
        debug.eupd = debug.getv0 = 4;
    }
    ~DebugAllOn() {
        using namespace arnoldi::detail;
        debug.aupd = aupd; debug.aup2 = aup2; debug.aitr = aitr;
        debug.eigh = eigh; debug.gets = gets; debug.apps = apps;
        debug.eupd = eupd; debug.getv0 = getv0_;
    }
};

// Cover the various `if (msglvl > 0/1/2/3)` debug branches inside *aitr,
// *aupd, sapps/napps, sgets/ngets, *eupd, and getv0 by running once with
// every debug knob turned all the way up.  Output is voluminous but only
// printed during this single test.
TEST_CASE("test_debug_branches", "[modes]") {
    DebugAllOn guard;
    {
        const int n = 10, nev = 2, ncv = 6;
        arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "LM", nev, ncv);
        s.tol(1e-8).maxiter(100);
        s.solve([n](const double* x, double* y) {
            y[0] = 2.0 * x[0] - x[1];
            for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
            y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
        });
        auto r = s.eigenpairs(true);
        auto exact = pick_real(tridiag_lap_spectrum(n), nev, "LM");
        double err = compare_real(r.values, exact);
        std::printf("    debug sym LM err = %.3e\n", err);
        check("debug sym LM matches exact", err < 1e-13);
    }
    {
        const int n = 10, nev = 2, ncv = 8;
        arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LM", nev, ncv);
        s.tol(1e-8).maxiter(100);
        s.solve([n](const double* x, double* y) {
            y[0] = 2.0 * x[0] - x[1];
            for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
            y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
        });
        auto r = s.eigenpairs(true);
        std::vector<double> got(r.values_re.begin(), r.values_re.begin() + nev);
        auto exact = pick_real(tridiag_lap_spectrum(n), nev, "LM");
        double err = compare_real(got, exact);
        std::printf("    debug nonsym LM err = %.3e\n", err);
        check("debug nonsym LM matches exact", err < 1e-13);
    }
    {
        const int n = 12, nev = 4, ncv = 10;
        arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "BE", nev, ncv);
        s.tol(1e-8).maxiter(200);
        s.solve([n](const double* x, double* y) {
            y[0] = 2.0 * x[0] - x[1];
            for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
            y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
        });
        auto r = s.eigenpairs(true);
        auto exact = pick_real(tridiag_lap_spectrum(n), nev, "BE");
        double err = compare_real(r.values, exact);
        std::printf("    debug sym BE err = %.3e\n", err);
        check("debug sym BE matches exact", err < 1e-13);
    }
}

// Force the float specializations of detail::Ops (lascl_, lanhs, swap) by
// running both Sym<float> + which="BE" (BE branch swaps with sswap_) and
// Nonsym<float>.  Standard tridiagonal Laplacian, small problem size.
// Tolerances are looser than the double tests because of single precision.
TEST_CASE("test_float_overloads", "[modes]") {
    {
        const int n = 16, nev = 4, ncv = 12;
        arnoldi::Arnoldi<arnoldi::Kind::Sym, float> s("I", n, "BE", nev, ncv);
        s.tol(1e-4f).maxiter(300);
        s.solve([n](const float* x, float* y) {
            y[0] = 2.0f * x[0] - x[1];
            for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0f * x[i] - x[i + 1];
            y[n - 1] = -x[n - 2] + 2.0f * x[n - 1];
        });
        auto r = s.eigenpairs(true);
        std::vector<double> got(r.values.begin(), r.values.end());
        auto exact = pick_real(tridiag_lap_spectrum(n), nev, "BE");
        double err = compare_real(got, exact);
        std::printf("    sym<float> BE max rel err = %.3e\n", err);
        check("Sym<float> BE: matches exact spectrum", err < 1e-4);
    }
    {
        const int n = 16, nev = 4, ncv = 12;
        arnoldi::Arnoldi<arnoldi::Kind::Nonsym, float> s("I", n, "LM", nev, ncv);
        s.tol(1e-4f).maxiter(300);
        s.solve([n](const float* x, float* y) {
            y[0] = 2.0f * x[0] - x[1];
            for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0f * x[i] - x[i + 1];
            y[n - 1] = -x[n - 2] + 2.0f * x[n - 1];
        });
        auto r = s.eigenpairs(true);
        std::vector<double> got(r.values_re.begin(), r.values_re.begin() + nev);
        auto exact = pick_real(tridiag_lap_spectrum(n), nev, "LM");
        double err = compare_real(got, exact);
        std::printf("    nonsym<float> LM max rel err = %.3e\n", err);
        check("Nonsym<float> LM: matches exact spectrum", err < 1e-4);
    }
}

// Run Nonsym generalized (bmat='G') with a tight ncv so the iter loop in
// naup2 runs many times.  Each pass through the loop hits the bmat=='G'
// post-napps recompute that the more relaxed shift-invert tests skip
// because they converge in a single iter.
TEST_CASE("test_nonsym_generalized_many_iters", "[modes]") {
    const int n = 60, nev = 6, ncv = 10;
    const double sigma = 0.5;
    ShiftInvertFE shift(n, sigma);
    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("G", n, "LM", nev, ncv);
    s.tol(1e-12).maxiter(1000).mode(3).ishift(1);
    s.solve(
        [&](const double* x, double* y) { mv_mass(n, x, y); shift.solve(y); },
        [&](const double* x, double* y) { mv_mass(n, x, y); });
    check("Nonsym G many-iter: solve completed", s.info() >= 0);
    auto r = s.eigenpairs(true, sigma, 0.0);

    auto exact = pick_real(fe_lap_spectrum(n), nev, "NEAR", sigma);
    std::vector<double> got(r.values_re.begin(), r.values_re.begin() + nev);
    double err = compare_real(got, exact);
    std::printf("    nonsym G many-iter max rel err = %.3e\n", err);
    check("Nonsym G many-iter: matches exact FE spectrum", err < 1e-8);
}

// Sym mode 2 with intentionally tight ncv to force many iterations through
// saup2's main loop, hitting the bmat[0]=='G' post-sapps refresh and the
// debug-print branches gated by msglvl.  Operator is A x = λ M x with the
// dimensionless tridiagonal A = [-1, 2, -1] and mass M = [1, 4, 1] / 6;
// the analytic spectrum is the FE Laplacian's with h ≡ 1 (no scaling):
//   λ_k = 6 (1 - cos(kπ/(n+1))) / (2 + cos(kπ/(n+1))).
TEST_CASE("test_sym_generalized_many_iters", "[modes]") {
    const int n = 60, nev = 4, ncv = 8;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("G", n, "LM", nev, ncv);
    s.tol(1e-12).maxiter(1000).mode(2);
    auto av = [n](const double* x, double* y) {
        y[0] = 2.0 * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
        y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
    };
    auto mv = [n](const double* x, double* y) {
        y[0] = (4.0 * x[0] + x[1]) / 6.0;
        for (int i = 1; i < n - 1; ++i) y[i] = (x[i - 1] + 4.0 * x[i] + x[i + 1]) / 6.0;
        y[n - 1] = (x[n - 2] + 4.0 * x[n - 1]) / 6.0;
    };
    auto solve_M = [n](double* y) {
        std::vector<double> dl(n - 1, 1.0 / 6.0), dd(n, 4.0 / 6.0), du(n - 1, 1.0 / 6.0);
        for (int i = 0; i < n - 1; ++i) {
            double f = dl[i] / dd[i]; dd[i + 1] -= f * du[i]; y[i + 1] -= f * y[i];
        }
        y[n - 1] /= dd[n - 1];
        for (int i = n - 2; i >= 0; --i) y[i] = (y[i] - du[i] * y[i + 1]) / dd[i];
    };
    // Mode-2 OP: y = M^-1 A x, AND overwrite x with A x (see test_sym_mode2_generalized).
    s.solve(
        [&](const double* x_in, double* y) {
            double* x = const_cast<double*>(x_in);
            std::vector<double> ax(n);
            ax[0] = 2.0 * x[0] - x[1];
            for (int i = 1; i < n - 1; ++i) ax[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
            ax[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
            std::copy(ax.begin(), ax.end(), x);
            std::copy(ax.begin(), ax.end(), y);
            solve_M(y);
        },
        mv);
    check("Sym G many-iter: solve completed", s.info() >= 0);
    auto r = s.eigenpairs(true);

    std::vector<double> full(n);
    for (int k = 1; k <= n; ++k) {
        double c = std::cos(k * M_PI / (n + 1));
        full[k - 1] = 6.0 * (1.0 - c) / (2.0 + c);
    }
    auto exact = pick_real(full, nev, "LM");
    double err = compare_real(r.values, exact);
    std::printf("    sym G many-iter max rel err = %.3e\n", err);
    check("Sym G many-iter: matches exact spectrum", err < 1e-9);
    (void)av;
}

// Force the REALPT (mode 3 with sigmai != 0) and IMAGPT (mode 4) branches in
// neupd's type-selection logic.
TEST_CASE("test_nonsym_neupd_realpt_imagpt", "[modes]") {
    const int n = 16, nev = 2, ncv = 8;
    auto av = [n](const double* x, double* y) {
        y[0] = x[1];
        for (int i = 1; i < n - 1; ++i) y[i] = x[i + 1] - x[i - 1];
        y[n - 1] = -x[n - 2];
    };
    {
        // mode=3 + nonzero sigma_i -> REALPT
        arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LM", nev, ncv);
        s.tol(1e-6).maxiter(300).mode(3);
        s.solve(av);
        auto r = s.eigenpairs(true, 0.0, 1.0);
        check("Nonsym mode3 sigma_i!=0 (REALPT): eigenpairs returned",
              r.values_re.size() == (size_t)(nev + 1));
    }
    {
        // mode=4 -> IMAGPT
        arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LM", nev, ncv);
        s.tol(1e-6).maxiter(300).mode(4);
        s.solve(av);
        auto r = s.eigenpairs(true, 0.5, 0.5);
        check("Nonsym mode4 (IMAGPT): eigenpairs returned",
              r.values_re.size() == (size_t)(nev + 1));
    }
}

// Antisymmetric tridiagonal: y[0]=x[1]; y[i]=x[i+1]-x[i-1]; y[n-1]=-x[n-2].
// Spectrum is ±2i cos(kπ/(n+1)) so neupd's conjugate-pair post-processing
// branches (REGULR & SHIFTI lapy2/scal blocks) get exercised. Real parts must
// be ≈ 0 and imaginary magnitudes must match the exact closed form.
TEST_CASE("test_nonsym_complex_eigenvalues", "[modes]") {
    const int n = 16, nev = 4, ncv = 12;
    auto av = [n](const double* x, double* y) {
        y[0] = x[1];
        for (int i = 1; i < n - 1; ++i) y[i] = x[i + 1] - x[i - 1];
        y[n - 1] = -x[n - 2];
    };
    auto exact_im = skew_tridiag_imag_spectrum(n);  // signed; |.| gives magnitude
    {
        // which="LM" returns the nev largest |λ|.  Take nev exact magnitudes.
        arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LM", nev, ncv);
        s.tol(1e-10).maxiter(500);
        s.solve(av);
        auto r = s.eigenpairs(true);
        std::vector<double> got_im(r.values_im.begin(), r.values_im.begin() + nev);
        std::vector<double> got_re(r.values_re.begin(), r.values_re.begin() + nev);
        auto exact_top = pick_real(exact_im, nev, "LM");
        double err = compare_abs(got_im, exact_top);
        double max_re = 0.0;
        for (double v : got_re) max_re = std::max(max_re, std::abs(v));
        std::printf("    nonsym complex LM max rel err = %.3e, max |re| = %.3e\n", err, max_re);
        check("Nonsym complex LM: real parts ~ 0", max_re < 1e-9);
        check("Nonsym complex LM: matches |2i cos(kπ/(n+1))|", err < 1e-9);
    }
    {
        // which="LI" with a real matrix: complex conjugate pairs are picked
        // together, so for nev=4 we get the 2 largest |Im| as ±α₁, ±α₂.
        // Compare on magnitudes since signs alternate within each pair.
        arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LI", nev, ncv);
        s.tol(1e-10).maxiter(500);
        s.solve(av);
        auto r = s.eigenpairs(false);
        std::vector<double> got_im(r.values_im.begin(), r.values_im.begin() + nev);
        auto exact_top = pick_real(exact_im, nev, "LM");
        double err = compare_abs(got_im, exact_top);
        std::printf("    nonsym complex LI max rel err = %.3e\n", err);
        check("Nonsym complex LI rvec=false: matches exact imag pairs", err < 1e-9);
    }
}
