// Drive the low-level saupd/naupd/seupd/neupd validation branches that the
// high-level Arnoldi<> constructor catches before they're reached. Each call
// is constructed to trip exactly one ierr return value so the corresponding
// branch in the *aupd/*eupd dispatchers is executed.

#include <arnoldi/arnoldi.hpp>
#include <arnoldi/comm.hpp>
#include <arnoldi/detail/nonsym.hpp>
#include <arnoldi/detail/sym.hpp>

#include <array>
#include <complex>
#include <cstdio>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void check(const char* name, bool cond, int got = 0, int want = 0) {
    if (cond) {
        std::printf("  OK    %s\n", name);
        ++g_pass;
    } else {
        std::printf("  FAIL  %s (got info=%d, want=%d)\n", name, got, want);
        ++g_fail;
    }
}

// Convenience holder for a "valid" workspace; individual fields are then
// adjusted to provoke a specific ierr return.
struct SaupdHarness {
    int n = 16, nev = 2, ncv = 6;
    int lworkl;
    double tol = 0.0;
    int info = 0;
    std::vector<double> resid, v, workd, workl;
    std::array<int, 11> iparam{};
    std::array<int, 14> ipntr{};
    arnoldi::SerialComm comm;

    SaupdHarness() {
        lworkl = ncv * ncv + 8 * ncv;
        resid.assign(n, 0.0);
        v.assign(n * ncv, 0.0);
        workd.assign(3 * n, 0.0);
        workl.assign(lworkl, 0.0);
        iparam[0] = 1;     // ishift
        iparam[2] = 100;   // mxiter
        iparam[6] = 1;     // mode
    }

    int run() {
        info = 0;
        arnoldi::detail::saupd<double>(
            "I", n, "LM", nev, tol, resid.data(), ncv, v.data(), n,
            iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl, info,
            [](const double*, double*) {}, [](const double*, double*) {}, comm);
        return info;
    }
};

struct NaupdHarness {
    int n = 16, nev = 2, ncv = 8;
    int lworkl;
    double tol = 0.0;
    int info = 0;
    std::vector<double> resid, v, workd, workl;
    std::array<int, 11> iparam{};
    std::array<int, 14> ipntr{};
    arnoldi::SerialComm comm;

    NaupdHarness() {
        lworkl = 3 * ncv * ncv + 6 * ncv;
        resid.assign(n, 0.0);
        v.assign(n * ncv, 0.0);
        workd.assign(3 * n, 0.0);
        workl.assign(lworkl, 0.0);
        iparam[0] = 1;
        iparam[2] = 100;
        iparam[6] = 1;
    }

    int run() {
        info = 0;
        arnoldi::detail::naupd<double>(
            "I", n, "LM", nev, tol, resid.data(), ncv, v.data(), n,
            iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl, info,
            [](const double*, double*) {}, [](const double*, double*) {}, comm);
        return info;
    }
};

static void test_saupd_errors() {
    {
        SaupdHarness h; h.n = 0;
        check("saupd n<=0  -> -1", h.run() == -1, h.info, -1);
    }
    {
        SaupdHarness h; h.nev = 0;
        check("saupd nev<=0 -> -2", h.run() == -2, h.info, -2);
    }
    {
        SaupdHarness h; h.ncv = h.nev;
        check("saupd ncv<=nev -> -3", h.run() == -3, h.info, -3);
    }
    {
        SaupdHarness h; h.iparam[2] = 0;
        check("saupd mxiter<=0 -> -4", h.run() == -4, h.info, -4);
    }
    {
        // saupd run() hardcodes which="LM". Build a custom call with bad which.
        SaupdHarness h; arnoldi::SerialComm c;
        arnoldi::detail::saupd<double>(
            "I", h.n, "ZZ", h.nev, h.tol, h.resid.data(), h.ncv, h.v.data(), h.n,
            h.iparam.data(), h.ipntr.data(), h.workd.data(), h.workl.data(), h.lworkl, h.info,
            [](const double*, double*) {}, [](const double*, double*) {}, c);
        check("saupd which invalid -> -5", h.info == -5, h.info, -5);
    }
    {
        SaupdHarness h; arnoldi::SerialComm c;
        arnoldi::detail::saupd<double>(
            "X", h.n, "LM", h.nev, h.tol, h.resid.data(), h.ncv, h.v.data(), h.n,
            h.iparam.data(), h.ipntr.data(), h.workd.data(), h.workl.data(), h.lworkl, h.info,
            [](const double*, double*) {}, [](const double*, double*) {}, c);
        check("saupd bmat invalid -> -6", h.info == -6, h.info, -6);
    }
    {
        SaupdHarness h; h.lworkl = 1;
        check("saupd lworkl small -> -7", h.run() == -7, h.info, -7);
    }
    {
        SaupdHarness h; h.iparam[6] = 0;
        check("saupd mode invalid -> -10", h.run() == -10, h.info, -10);
    }
    {
        // mode 1 with bmat='G' is illegal.
        SaupdHarness h; arnoldi::SerialComm c;
        h.iparam[6] = 1;
        arnoldi::detail::saupd<double>(
            "G", h.n, "LM", h.nev, h.tol, h.resid.data(), h.ncv, h.v.data(), h.n,
            h.iparam.data(), h.ipntr.data(), h.workd.data(), h.workl.data(), h.lworkl, h.info,
            [](const double*, double*) {}, [](const double*, double*) {}, c);
        check("saupd mode=1 + bmat=G -> -11", h.info == -11, h.info, -11);
    }
    {
        SaupdHarness h; h.iparam[0] = 2;
        check("saupd ishift invalid -> -12", h.run() == -12, h.info, -12);
    }
}

static void test_naupd_errors() {
    {
        NaupdHarness h; h.n = 0;
        check("naupd n<=0  -> -1", h.run() == -1, h.info, -1);
    }
    {
        NaupdHarness h; h.nev = 0;
        check("naupd nev<=0 -> -2", h.run() == -2, h.info, -2);
    }
    {
        // Nonsym requires ncv > nev+1.
        NaupdHarness h; h.ncv = h.nev + 1;
        check("naupd ncv<=nev+1 -> -3", h.run() == -3, h.info, -3);
    }
    {
        NaupdHarness h; h.iparam[2] = 0;
        check("naupd mxiter<=0 -> -4", h.run() == -4, h.info, -4);
    }
    {
        NaupdHarness h; arnoldi::SerialComm c;
        arnoldi::detail::naupd<double>(
            "I", h.n, "ZZ", h.nev, h.tol, h.resid.data(), h.ncv, h.v.data(), h.n,
            h.iparam.data(), h.ipntr.data(), h.workd.data(), h.workl.data(), h.lworkl, h.info,
            [](const double*, double*) {}, [](const double*, double*) {}, c);
        check("naupd which invalid -> -5", h.info == -5, h.info, -5);
    }
    {
        NaupdHarness h; arnoldi::SerialComm c;
        arnoldi::detail::naupd<double>(
            "X", h.n, "LM", h.nev, h.tol, h.resid.data(), h.ncv, h.v.data(), h.n,
            h.iparam.data(), h.ipntr.data(), h.workd.data(), h.workl.data(), h.lworkl, h.info,
            [](const double*, double*) {}, [](const double*, double*) {}, c);
        check("naupd bmat invalid -> -6", h.info == -6, h.info, -6);
    }
    {
        NaupdHarness h; h.lworkl = 1;
        check("naupd lworkl small -> -7", h.run() == -7, h.info, -7);
    }
    {
        NaupdHarness h; h.iparam[6] = 0;
        check("naupd mode invalid -> -10", h.run() == -10, h.info, -10);
    }
    {
        NaupdHarness h; arnoldi::SerialComm c;
        h.iparam[6] = 1;
        arnoldi::detail::naupd<double>(
            "G", h.n, "LM", h.nev, h.tol, h.resid.data(), h.ncv, h.v.data(), h.n,
            h.iparam.data(), h.ipntr.data(), h.workd.data(), h.workl.data(), h.lworkl, h.info,
            [](const double*, double*) {}, [](const double*, double*) {}, c);
        check("naupd mode=1 + bmat=G -> -11", h.info == -11, h.info, -11);
    }
    {
        NaupdHarness h; h.iparam[0] = 2;
        check("naupd ishift invalid -> -12", h.run() == -12, h.info, -12);
    }
}

// seupd validates its arguments separately; drive several of those branches.
// seupd assumes nconv>0 (read from iparam[4]); pre-set it so other checks fire.
static void test_seupd_errors() {
    SaupdHarness base;
    base.iparam[4] = 1;  // nconv

    auto run_seupd = [&](const char* bmat, const char* which, int n, int nev, int ncv,
                         int lworkl, int mode, bool rvec, const char* howmny,
                         std::array<int,11>& iparam, std::array<int,14>& ipntr,
                         double* resid, double* v, double* workd, double* workl) -> int {
        std::vector<double> d(std::max(nev, 1));
        std::vector<double> z(std::max(n * std::max(nev,1), 1));
        int info = 0;
        iparam[6] = mode;
        arnoldi::detail::seupd<double>(
            rvec, howmny, d.data(), rvec ? z.data() : nullptr,
            n, 0.0, bmat, n, which, nev, 0.0, resid, ncv, v, n,
            iparam.data(), ipntr.data(), workd, workl, lworkl, info,
            arnoldi::SerialComm{});
        return info;
    };

    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("I", "LM", 0, h.nev, h.ncv, h.lworkl, 1, false, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd n<=0 -> -1", info == -1, info, -1);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("I", "LM", h.n, 0, h.ncv, h.lworkl, 1, false, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd nev<=0 -> -2", info == -2, info, -2);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("I", "LM", h.n, h.nev, h.nev, h.lworkl, 1, false, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd ncv<=nev -> -3", info == -3, info, -3);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("I", "ZZ", h.n, h.nev, h.ncv, h.lworkl, 1, false, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd which invalid -> -5", info == -5, info, -5);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("X", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, false, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd bmat invalid -> -6", info == -6, info, -6);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("I", "LM", h.n, h.nev, h.ncv, 1, 1, true, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd rvec & lworkl small -> -7", info == -7, info, -7);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("I", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, true, "Z",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd howmny invalid -> -15", info == -15, info, -15);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("I", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, true, "S",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd howmny='S' -> -16", info == -16, info, -16);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("I", "LM", h.n, h.nev, h.ncv, h.lworkl, 99, false, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd mode invalid -> -10", info == -10, info, -10);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 1;
        int info = run_seupd("G", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, false, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd mode=1+bmat=G -> -11", info == -11, info, -11);
    }
    {
        SaupdHarness h = base; h.iparam[4] = 0;  // nconv==0 returns silently with info=0
        int info = run_seupd("I", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, false, "A",
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("seupd nconv==0 returns ok", info == 0, info, 0);
    }
}

static void test_neupd_errors() {
    NaupdHarness base;

    auto run_neupd = [&](const char* bmat, const char* which, int n, int nev, int ncv,
                         int lworkl, int mode, bool rvec, const char* howmny,
                         int nconv,
                         std::array<int,11>& iparam, std::array<int,14>& ipntr,
                         double* resid, double* v, double* workd, double* workl) -> int {
        const int dim = std::max(nev + 1, 1);
        std::vector<double> dr(dim), di(dim);
        std::vector<double> z(std::max(n * dim, 1));
        std::vector<double> workev(std::max(3 * ncv, 1));
        int info = 0;
        iparam[6] = mode;
        iparam[4] = nconv;
        arnoldi::detail::neupd<double>(
            rvec, howmny, dr.data(), di.data(), rvec ? z.data() : nullptr,
            n, 0.0, 0.0, workev.data(),
            bmat, n, which, nev, 0.0, resid, ncv, v, n,
            iparam.data(), ipntr.data(), workd, workl, lworkl, info);
        return info;
    };

    {
        NaupdHarness h = base;
        int info = run_neupd("I", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, false, "A", 0,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd nconv<=0 -> -14", info == -14, info, -14);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("I", "LM", 0, h.nev, h.ncv, h.lworkl, 1, false, "A", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd n<=0 -> -1", info == -1, info, -1);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("I", "LM", h.n, 0, h.ncv, h.lworkl, 1, false, "A", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd nev<=0 -> -2", info == -2, info, -2);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("I", "LM", h.n, h.nev, h.nev + 1, h.lworkl, 1, false, "A", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd ncv<=nev+1 -> -3", info == -3, info, -3);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("I", "ZZ", h.n, h.nev, h.ncv, h.lworkl, 1, false, "A", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd which invalid -> -5", info == -5, info, -5);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("X", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, false, "A", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd bmat invalid -> -6", info == -6, info, -6);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("I", "LM", h.n, h.nev, h.ncv, 1, 1, false, "A", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd lworkl small -> -7", info == -7, info, -7);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("I", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, true, "Z", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd howmny invalid -> -13", info == -13, info, -13);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("I", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, true, "S", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd howmny='S' -> -12", info == -12, info, -12);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("I", "LM", h.n, h.nev, h.ncv, h.lworkl, 99, false, "A", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd mode invalid -> -10", info == -10, info, -10);
    }
    {
        NaupdHarness h = base;
        int info = run_neupd("G", "LM", h.n, h.nev, h.ncv, h.lworkl, 1, false, "A", 1,
                             h.iparam, h.ipntr, h.resid.data(), h.v.data(), h.workd.data(), h.workl.data());
        check("neupd mode=1+bmat=G -> -11", info == -11, info, -11);
    }
}

// Drive a real solve all the way through saupd, then call seupd directly with
// z aliasing v.  Forces the seupd `if (z == v) tmpbuf` allocation branch
// which the high-level Arnoldi class never triggers (it owns separate buffers).
static void test_seupd_aliased_z_v() {
    const int n = 16, nev = 3, ncv = 8;
    int   lworkl = ncv * (ncv + 8);
    double tol   = 0.0;
    int   info   = 0;
    std::vector<double> resid(n, 0.0), v(n * ncv, 0.0), workd(3 * n, 0.0), workl(lworkl, 0.0);
    std::array<int, 11> iparam{};
    std::array<int, 14> ipntr{};
    iparam[0] = 1; iparam[2] = 200; iparam[6] = 1;
    arnoldi::SerialComm comm;

    auto op = [n](const double* x, double* y) {
        y[0] = 2.0 * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
        y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
    };
    auto bop = [](const double*, double*) {};

    arnoldi::detail::saupd<double>("I", n, "LM", nev, tol, resid.data(), ncv, v.data(), n,
                                   iparam.data(), ipntr.data(), workd.data(), workl.data(),
                                   lworkl, info, op, bop, comm);
    check("aliased seupd: saupd ok", info >= 0, info, 0);
    if (info < 0) return;

    std::vector<double> dvals(nev, 0.0);
    int seinfo = 0;
    arnoldi::detail::seupd<double>(true, "A", dvals.data(),
                                   v.data(), n,  // z == v aliased
                                   0.0,
                                   "I", n, "LM", nev, tol, resid.data(), ncv,
                                   v.data(), n, iparam.data(), ipntr.data(),
                                   workd.data(), workl.data(), lworkl, seinfo,
                                   comm);
    check("aliased seupd: returns ok", seinfo == 0, seinfo, 0);
}

// Call the no-bop, no-comm 3-arg saupd/naupd helper overloads directly.
// The high-level Arnoldi class always passes a bop lambda so these helper
// overloads are unreachable through the public API.
static void test_no_bop_overloads() {
    {
        const int n = 16, nev = 2, ncv = 8;
        int   lworkl = ncv * (ncv + 8);
        double tol   = 0.0;
        int   info   = 0;
        std::vector<double> resid(n, 0.0), v(n * ncv, 0.0), workd(3 * n, 0.0), workl(lworkl, 0.0);
        std::array<int, 11> iparam{};
        std::array<int, 14> ipntr{};
        iparam[0] = 1; iparam[2] = 200; iparam[6] = 1;

        arnoldi::detail::saupd<double>(
            "I", n, "LM", nev, tol, resid.data(), ncv, v.data(), n,
            iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl, info,
            [n](const double* x, double* y) {
                y[0] = 2.0 * x[0] - x[1];
                for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
                y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
            });
        check("no-bop saupd<double>: ok", info >= 0, info, 0);
    }
    {
        const int n = 16, nev = 2, ncv = 8;
        int   lworkl = 3 * ncv * ncv + 6 * ncv;
        double tol   = 0.0;
        int   info   = 0;
        std::vector<double> resid(n, 0.0), v(n * ncv, 0.0), workd(3 * n, 0.0), workl(lworkl, 0.0);
        std::array<int, 11> iparam{};
        std::array<int, 14> ipntr{};
        iparam[0] = 1; iparam[2] = 200; iparam[6] = 1;

        arnoldi::detail::naupd<double>(
            "I", n, "LM", nev, tol, resid.data(), ncv, v.data(), n,
            iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl, info,
            [n](const double* x, double* y) {
                y[0] = 2.0 * x[0] - x[1];
                for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
                y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
            });
        check("no-bop naupd<double>: ok", info >= 0, info, 0);
    }
}

// User-supplied initial residual: covers info_in_=1 path through naupd.
static void test_user_initial_residual_nonsym() {
    const int n = 32, nev = 3, ncv = 12;
    std::vector<double> resid(n, 1.0);

    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LM", nev, ncv);
    s.tol(1e-10).maxiter(500).initial_resid(resid.data());
    s.solve([n](const double* x, double* y) {
        y[0] = 2.0 * x[0] - x[1];
        for (int i = 1; i < n - 1; ++i) y[i] = -x[i - 1] + 2.0 * x[i] - x[i + 1];
        y[n - 1] = -x[n - 2] + 2.0 * x[n - 1];
    });
    check("Nonsym user initial resid: solve ok", s.info() >= 0);
}

int main() {
    std::printf("test_low_level_validation:\n");
    test_saupd_errors();
    test_naupd_errors();
    test_seupd_errors();
    test_neupd_errors();
    test_user_initial_residual_nonsym();
    test_seupd_aliased_z_v();
    test_no_bop_overloads();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
