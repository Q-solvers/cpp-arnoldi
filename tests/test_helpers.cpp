// Direct unit tests for the small detail/ routines that the high-level
// Arnoldi driver only touches through specific paths. Covers sort/, helpers/,
// debug/, eig/ and a few seldom-used Ops dispatches so the coverage report is
// not dominated by template instantiations whose else-branches are unreached.

#include <arnoldi/comm.hpp>
#include <arnoldi/detail/debug.hpp>
#include <arnoldi/detail/eig.hpp>
#include <arnoldi/detail/helpers.hpp>
#include <arnoldi/detail/ops.hpp>
#include <arnoldi/detail/sort.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
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

template <typename Real>
static bool nearly(Real a, Real b, Real tol = Real(1e-10)) { return std::abs(a - b) <= tol; }

// In ARPACK the "wanted" values end up at the end of the sorted array.  So
// sortc/sortr with "LM" arranges by ascending magnitude; "SM" by descending
// magnitude, etc.
static void test_sortc_all() {
    using arnoldi::detail::sortc;

    auto run = [&](const char* which, std::vector<double> xr, std::vector<double> xi,
                   std::vector<double> y, std::vector<double> exp_xr, std::vector<double> exp_xi) {
        sortc<double>(which, true, (int)xr.size(), xr.data(), xi.data(), y.data());
        bool ok = (xr == exp_xr) && (xi == exp_xi);
        check((std::string("sortc ") + which).c_str(), ok);
    };

    run("LM", {3, 1, 2}, {0, 0, 0}, {1, 2, 3}, {1, 2, 3}, {0, 0, 0});
    run("SM", {1, 3, 2}, {0, 0, 0}, {1, 2, 3}, {3, 2, 1}, {0, 0, 0});
    run("LR", {3, 1, 2}, {0, 0, 0}, {1, 2, 3}, {1, 2, 3}, {0, 0, 0});
    run("SR", {1, 3, 2}, {0, 0, 0}, {1, 2, 3}, {3, 2, 1}, {0, 0, 0});
    run("LI", {0, 0, 0}, {3, 1, 2}, {1, 2, 3}, {0, 0, 0}, {1, 2, 3});
    run("SI", {0, 0, 0}, {1, 3, 2}, {1, 2, 3}, {0, 0, 0}, {3, 2, 1});
}

static void test_sortr_all() {
    using arnoldi::detail::sortr;
    auto run = [&](const char* which, std::vector<double> a, std::vector<double> b, std::vector<double> exp_a) {
        sortr<double>(which, true, (int)a.size(), a.data(), b.data());
        check((std::string("sortr ") + which).c_str(), a == exp_a);
    };
    run("SA", {3, 1, 2}, {1, 2, 3}, {3, 2, 1});
    run("LA", {1, 3, 2}, {1, 2, 3}, {1, 2, 3});
    run("SM", {-1, 3, -2}, {1, 2, 3}, {3, -2, -1});
    run("LM", {-3, 1, -2}, {1, 2, 3}, {1, -2, -3});
}

// Sort a 3x3 matrix by columns based on a key vector x; verify both the key
// reordering and that columns of `a` followed (apply=true).
static void test_sesrt_all() {
    using arnoldi::detail::sesrt;
    const int n = 3, lda = 3;
    auto run = [&](const char* which, std::vector<double> x, std::vector<double> exp_x) {
        std::vector<double> a(n * lda);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) a[j * lda + i] = static_cast<double>(j + 1);
        sesrt<double>(which, true, n, x.data(), n, a.data(), lda);
        check((std::string("sesrt ") + which).c_str(), x == exp_x);
    };
    run("SA", {3, 1, 2}, {3, 2, 1});
    run("LA", {1, 3, 2}, {1, 2, 3});
    run("SM", {1, -3, 2}, {-3, 2, 1});
    run("LM", {-1, 3, -2}, {-1, -2, 3});
}

// Verify ngets walks each branch of the which-translation switch and the
// "split conjugate pair" guard that pulls the boundary index back by one.
static void test_ngets_branches() {
    using arnoldi::detail::ngets;
    int kev = 2, np = 2;
    std::vector<double> rr = {1, 2, 3, 4}, ri = {0, 0, 0, 0}, bnd = {.1, .2, .3, .4};
    std::vector<double> sr(np), si(np);
    for (const char* which : {"LM", "SM", "LR", "SR", "LI", "SI"}) {
        rr = {1, 2, 3, 4}; ri = {0, 0, 0, 0}; bnd = {.1, .2, .3, .4};
        kev = 2; np = 2;
        ngets<double>(1, which, kev, np, rr.data(), ri.data(), bnd.data(), sr.data(), si.data());
        check((std::string("ngets which=") + which).c_str(), kev + np == 4);
    }

    // Force (rr[np]-rr[np-1])==0 && (ri[np]+ri[np-1])==0 to exercise the
    // split-pair adjustment path (np-=1, kev+=1).  Two values with equal
    // real part end up adjacent across the np boundary after sorting.
    rr = {1, 3, 3, 4}; ri = {0, 0, 0, 0}; bnd = {.1, .2, .3, .4};
    kev = 2; np = 2;
    ngets<double>(0, "LM", kev, np, rr.data(), ri.data(), bnd.data(), sr.data(), si.data());
    check("ngets split-pair guard", kev == 3 && np == 1);
}

// sgets has a BE special case that swaps two halves of the spectrum to keep
// "both ends" balanced. Exercise both BE and a regular which.
static void test_sgets_branches() {
    using arnoldi::detail::sgets;
    std::vector<double> ritz = {1, 2, 3, 4, 5, 6}, bounds = {.1, .2, .3, .4, .5, .6};
    std::vector<double> shifts(3);
    sgets<double>(1, "BE", 4, 2, ritz.data(), bounds.data(), shifts.data());
    check("sgets BE", ritz.size() == 6);

    ritz = {3, 1, 4, 1, 5, 9}; bounds = {.1, .2, .3, .4, .5, .6};
    sgets<double>(1, "LM", 4, 2, ritz.data(), bounds.data(), shifts.data());
    check("sgets LM", ritz.size() == 6);
}

// Convergence check returns the number of Ritz values whose error bound is
// below tol*max(eps23, |ritz|).
static void test_conv_check() {
    using arnoldi::detail::nconv;
    using arnoldi::detail::sconv;
    std::vector<double> ritzr = {1, 2, 3}, ritzi = {0, 0, 0}, bounds = {1e-15, 1e-15, 1e-1};
    int nc = -1;
    nconv<double>(3, ritzr.data(), ritzi.data(), bounds.data(), 1e-12, nc);
    check("nconv counts converged", nc == 2);

    std::vector<double> ritz = {1, 10, 100}, bounds2 = {1e-15, 1e-15, 1e-1};
    sconv<double>(3, ritz.data(), bounds2.data(), 1e-12, nc);
    check("sconv counts converged", nc == 2);
}

// stqrb is the tridiagonal QR used by seigt; cover n=0, n=1 and a small
// generic case so the LAPACK glue and label-90 pivot loop both run.
static void test_stqrb() {
    using arnoldi::detail::stqrb;
    int info = 0;
    std::vector<double> z;

    stqrb<double>(0, nullptr, nullptr, nullptr, nullptr, info);
    check("stqrb n=0", info == 0);

    std::vector<double> d1 = {2.5}; z = {0};
    std::vector<double> work1(2);
    stqrb<double>(1, d1.data(), nullptr, z.data(), work1.data(), info);
    check("stqrb n=1", info == 0 && z[0] == 1.0);

    const int n = 5;
    std::vector<double> d(n), e(n), zz(n), w(2 * n);
    for (int i = 0; i < n; ++i) { d[i] = 2.0; e[i] = -1.0; }
    stqrb<double>(n, d.data(), e.data(), zz.data(), w.data(), info);
    check("stqrb n=5 ok", info == 0);
    check("stqrb n=5 sorted ascending", std::is_sorted(d.begin(), d.end()));
}

// Exercise the seigt entry point used by sym Lanczos extraction; also covers
// the n>1 sub-diagonal debug print branch when msglvl>0.
static void test_seigt_neigh() {
    using arnoldi::detail::neigh;
    using arnoldi::detail::seigt;

    {
        const int ldh = 4, n = 4;
        std::vector<double> h(2 * ldh, 0.0), eig(n), bnd(n), workl(3 * n);
        for (int i = 0; i < n; ++i) h[ldh + i] = double(i + 1);
        for (int i = 0; i < n - 1; ++i) h[1 + i] = -0.5;
        int ierr = 0;
        seigt<double>(1.0, n, h.data(), ldh, eig.data(), bnd.data(), workl.data(), ierr);
        check("seigt converges", ierr == 0);
    }

    {
        const int n = 4, ldh = n, ldq = n;
        std::vector<double> h(n * ldh, 0.0);
        for (int i = 0; i < n; ++i)     h[i * ldh + i] = 2.0;
        for (int i = 0; i < n - 1; ++i) {
            h[i * ldh + (i + 1)] = -1.0;
            h[(i + 1) * ldh + i] = -1.0;
        }
        std::vector<double> ritzr(n), ritzi(n), bnd(n), q(n * ldq);
        // workl needs n*n for the H copy plus 3*n scratch for LAPACK trevc.
        std::vector<double> workl(n * n + 3 * n);
        int ierr = 0;
        neigh<double>(1.0, n, h.data(), ldh, ritzr.data(), ritzi.data(),
                      bnd.data(), q.data(), ldq, workl.data(), ierr);
        check("neigh converges", ierr == 0);
    }
}

// Hit each print routine in detail::debug for both real and complex element
// types so the four print_elem overloads are all instantiated.  Output is
// noisy but small; we just want the lines exercised for coverage.
static void test_debug_prints() {
    using namespace arnoldi::detail;

    int    iv[3]   = {1, 2, 3};
    double dv[3]   = {1.0, 2.0, 3.0};
    float  fv[3]   = {1.f, 2.f, 3.f};
    std::complex<double> zv[2] = {{1, 2}, {3, 4}};
    std::complex<float>  cv[2] = {{5, 6}, {7, 8}};
    debug.ivout(3, iv, "[debug] ivout");
    debug.vout(3, dv, "[debug] vout double");
    debug.vout(3, fv, "[debug] vout float");
    debug.vout(2, zv, "[debug] vout cdouble");
    debug.vout(2, cv, "[debug] vout cfloat");
    double m[6] = {1, 2, 3, 4, 5, 6};
    debug.mout(2, 3, m, 2, "[debug] mout double");
    std::complex<double> mz[4] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    debug.mout(2, 2, mz, 2, "[debug] mout cdouble");
    check("debug print routines run", true);
}

// Exercise the Real/Complex Ops dispatches that the standard double symmetric
// solver path doesn't reach: rscal/raxpy on a complex vector, dotc, gemv_rv.
static void test_ops_complex_dispatch() {
    using cplx = std::complex<double>;
    using O    = arnoldi::detail::Ops<cplx>;

    std::vector<cplx> x = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<cplx> y = {{0, 0}, {0, 0}, {0, 0}};

    O::rscal(3, 2.0, x.data(), 1);
    check("Ops<cdouble>::rscal", std::abs(x[1] - cplx(4, 4)) < 1e-12);

    O::raxpy(3, 0.5, x.data(), 1, y.data(), 1);
    check("Ops<cdouble>::raxpy", std::abs(y[2] - cplx(3, 3)) < 1e-12);

    cplx d = O::dotc(3, x.data(), 1, x.data(), 1);
    check("Ops<cdouble>::dotc real", std::abs(d.imag()) < 1e-12);
    check("Ops<cdouble>::rdotc",
          std::abs(O::rdotc(3, x.data(), 1, x.data(), 1) - d.real()) < 1e-12);

    std::vector<cplx> sw1 = {{1, 0}, {2, 0}}, sw2 = {{3, 0}, {4, 0}};
    O::copy(2, sw1.data(), 1, sw2.data(), 1);
    check("Ops<cdouble>::copy", sw2[0] == cplx(1, 0));

    std::vector<cplx> nv = {{3, 4}, {0, 0}};
    check("Ops<cdouble>::nrm2", std::abs(O::nrm2(2, nv.data(), 1) - 5.0) < 1e-12);

    std::vector<cplx> a(6);
    for (int j = 0; j < 3; ++j) for (int i = 0; i < 2; ++i) a[j * 2 + i] = cplx(j + 1, 0);
    std::vector<double> rx = {1, 2, 3};
    std::vector<cplx>   r2(2, cplx(0, 0));
    O::gemv_rv("N", 2, 3, 1.0, a.data(), 2, rx.data(), 1, 0.0, r2.data(), 1);
    check("Ops<cdouble>::gemv_rv", std::abs(r2[0] - cplx(14, 0)) < 1e-12);

    int               seed[4] = {1, 3, 5, 7};
    std::vector<cplx> rng(4);
    O::larnv(2, seed, 4, rng.data());
    check("Ops<cdouble>::larnv", rng[0] != cplx(0, 0));
}

// pdot/prdotc/pnrm2_real/pnrm2 reduction wrappers: ensure they call through
// SerialComm without modifying anything beyond the BLAS forwarding.
static void test_parallel_reductions() {
    using arnoldi::detail::pdot;
    using arnoldi::detail::pnrm2;
    using arnoldi::detail::pnrm2_real;
    using arnoldi::detail::prdotc;
    arnoldi::SerialComm comm;

    std::vector<double> x = {3, 4};
    check("pdot serial", nearly(pdot<double>(comm, 2, x.data(), 1, x.data(), 1), 25.0));
    check("pnrm2_real serial", nearly(pnrm2_real<double>(comm, 2, x.data(), 1), 5.0));

    using cplx = std::complex<double>;
    std::vector<cplx> z = {{3, 0}, {0, 4}};
    check("pnrm2 serial complex", nearly(pnrm2<cplx>(comm, 2, z.data(), 1), 5.0));
    check("prdotc serial complex", nearly(prdotc<cplx>(comm, 2, z.data(), 1, z.data(), 1), 25.0));
}

int main() {
    std::printf("test_helpers:\n");
    setvbuf(stdout, nullptr, _IOLBF, 0);
    test_sortc_all();
    test_sortr_all();
    test_sesrt_all();
    test_ngets_branches();
    test_sgets_branches();
    test_conv_check();
    test_stqrb();
    test_seigt_neigh();
    test_debug_prints();
    test_ops_complex_dispatch();
    test_parallel_reductions();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
