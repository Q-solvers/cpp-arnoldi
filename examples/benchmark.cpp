// Performance + correctness benchmark: C++ arnoldi::Arnoldi vs Fortran ARPACK
//
// Runs 4 driver problems (dsdrv1, dndrv1, dsdrv2, dndrv2) at multiple
// problem sizes.  For each problem, both the C++ class API and the direct
// Fortran reverse-communication interface are timed and their eigenvalues
// compared.
//
// The Fortran path calls dsaupd_/dseupd_/dnaupd_/dneupd_ through
// extern "C" — they resolve to the bridge or to the real libarpack
// depending on what this binary is linked against.
//
// Usage: benchmark [--reps 10] [--warmup 3]

#include <arnoldi/arnoldi.hpp>
#include "kernels.hpp"
#include "lapack_extra.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Fortran ARPACK entry points (resolved at link time).
extern "C" {
void dsaupd_(int* ido, const char* bmat, const int* n, const char* which,
             const int* nev, const double* tol, double* resid, const int* ncv,
             double* v, const int* ldv, int* iparam, int* ipntr,
             double* workd, double* workl, const int* lworkl, int* info);
void dseupd_(const int* rvec, const char* howmny, int* select, double* d,
             double* z, const int* ldz, const double* sigma,
             const char* bmat, const int* n, const char* which,
             const int* nev, const double* tol, double* resid, const int* ncv,
             double* v, const int* ldv, int* iparam, int* ipntr,
             double* workd, double* workl, const int* lworkl, int* info);
void dnaupd_(int* ido, const char* bmat, const int* n, const char* which,
             const int* nev, const double* tol, double* resid, const int* ncv,
             double* v, const int* ldv, int* iparam, int* ipntr,
             double* workd, double* workl, const int* lworkl, int* info);
void dneupd_(const int* rvec, const char* howmny, int* select, double* dr,
             double* di, double* z, const int* ldz, const double* sigmar,
             const double* sigmai, double* workev,
             const char* bmat, const int* n, const char* which,
             const int* nev, const double* tol, double* resid, const int* ncv,
             double* v, const int* ldv, int* iparam, int* ipntr,
             double* workd, double* workl, const int* lworkl, int* info);
void dcopy_(const int* n, const double* x, const int* incx,
            double* y, const int* incy);
}

using Clock = std::chrono::high_resolution_clock;
using Eig = std::pair<double, double>;  // (real, imag)

struct Timing {
    long long median_us, min_us, max_us;
};

template<class Fn>
static Timing time_fn(Fn&& fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<long long> t(reps);
    for (int i = 0; i < reps; ++i) {
        auto a = Clock::now();
        fn();
        auto b = Clock::now();
        t[i] = std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    }
    std::sort(t.begin(), t.end());
    return {t[reps / 2], t.front(), t.back()};
}

// Eigenvalue comparison via unordered-set matching by magnitude.
static double eig_max_error(std::vector<Eig> a, std::vector<Eig> b) {
    auto mag = [](Eig z) { return std::sqrt(z.first * z.first + z.second * z.second); };
    auto cmp = [&](const Eig& x, const Eig& y) { return mag(x) < mag(y); };
    std::sort(a.begin(), a.end(), cmp);
    std::sort(b.begin(), b.end(), cmp);
    double max_err = 0.0;
    int m = std::min(a.size(), b.size());
    for (int i = 0; i < m; ++i) {
        double dr = a[i].first - b[i].first;
        double di = a[i].second - b[i].second;
        double err = std::sqrt(dr * dr + di * di);
        double scale = std::max(1e-14, mag(a[i]));
        max_err = std::max(max_err, err / scale);
    }
    return max_err;
}

using arnoldi_examples::av_sym_laplacian_2d;
using arnoldi_examples::av_conv_diff_2d;
using arnoldi_examples::av_laplacian_1d;

// C++ solvers (return converged eigenvalues).

static std::vector<Eig> cpp_dsdrv1(int nx) {
    int n = nx * nx, nev = 4, ncv = std::min(20, n - 1);
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "SM", nev, ncv);
    s.tol(0.0).maxiter(300).mode(1).ishift(1);
    s.solve([&](const double* x, double* y) { av_sym_laplacian_2d(nx, n, x, y); });
    auto r = s.eigenpairs(true, 0.0);
    std::vector<Eig> out;
    for (int i = 0; i < s.num_converged(); ++i)
        out.push_back({r.values[i], 0.0});
    return out;
}

static std::vector<Eig> cpp_dndrv1(int nx) {
    int n = nx * nx, nev = 4, ncv = std::min(20, n - 1);
    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "SM", nev, ncv);
    s.tol(0.0).maxiter(300).mode(1).ishift(1);
    s.solve([&](const double* x, double* y) { av_conv_diff_2d(nx, n, 0.0, x, y); });
    auto r = s.eigenpairs(true, 0.0, 0.0);
    std::vector<Eig> out;
    for (int i = 0; i < s.num_converged(); ++i)
        out.push_back({r.values_re[i], r.values_im[i]});
    return out;
}

static std::vector<Eig> cpp_dsdrv2(int n) {
    int nev = 4, ncv = std::min(20, n - 1);
    double sigma = 0.0;
    double h2 = 1.0 / ((n + 1.0) * (n + 1.0));
    std::vector<double> ad(n, 2.0 / h2 - sigma), adl(n, -1.0 / h2),
                        adu(n, -1.0 / h2), adu2(n, 0.0);
    std::vector<int> ipiv(n);
    arnoldi_examples::lapackx::gttrf<double>(n, adl.data(), ad.data(),
                                             adu.data(), adu2.data(), ipiv.data());
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "LM", nev, ncv);
    s.tol(0.0).maxiter(300).mode(3).ishift(1);
    s.solve([&](const double* x, double* y) {
        std::copy(x, x + n, y);
        arnoldi_examples::lapackx::gttrs<double>("N", n, 1, adl.data(), ad.data(),
                                                 adu.data(), adu2.data(), ipiv.data(), y, n);
    });
    auto r = s.eigenpairs(true, sigma);
    std::vector<Eig> out;
    for (int i = 0; i < s.num_converged(); ++i)
        out.push_back({r.values[i], 0.0});
    return out;
}

static std::vector<Eig> cpp_dndrv2(int n) {
    int nev = 4, ncv = std::min(20, n - 1);
    double sigmar = 1.0, sigmai = 0.0, rho = 10.0;
    double h = 1.0 / (n + 1), ss = rho * h / 2.0;
    std::vector<double> dd(n, 2.0 - sigmar), dl(n, -1.0 - ss),
                        du(n, -1.0 + ss), du2(n, 0.0);
    std::vector<int> ipiv(n);
    arnoldi_examples::lapackx::gttrf<double>(n, dl.data(), dd.data(),
                                             du.data(), du2.data(), ipiv.data());
    arnoldi::Arnoldi<arnoldi::Kind::Nonsym, double> s("I", n, "LM", nev, ncv);
    s.tol(0.0).maxiter(300).mode(3).ishift(1);
    s.solve([&](const double* x, double* y) {
        std::copy(x, x + n, y);
        arnoldi_examples::lapackx::gttrs<double>("N", n, 1, dl.data(), dd.data(),
                                                 du.data(), du2.data(), ipiv.data(), y, n);
    });
    auto r = s.eigenpairs(true, sigmar, sigmai);
    std::vector<Eig> out;
    for (int i = 0; i < s.num_converged(); ++i)
        out.push_back({r.values_re[i], r.values_im[i]});
    return out;
}

// Fortran solvers (return converged eigenvalues).

static std::vector<Eig> f90_dsdrv1(int nx) {
    int n = nx * nx, nev = 4, ncv = std::min(20, n - 1);
    int ldv = n, lworkl = ncv * (ncv + 8);
    double tol = 0.0, sigma = 0.0;
    std::vector<double> v(ldv * ncv), workd(3 * n), workl(lworkl), resid(n), d(ncv * 2);
    std::vector<int> sel(ncv);
    int iparam[11] = {}, ipntr[11] = {};
    iparam[0] = 1; iparam[2] = 300; iparam[6] = 1;
    int ido = 0, info = 0;
    while (true) {
        dsaupd_(&ido, "I", &n, "SM", &nev, &tol, resid.data(), &ncv,
                v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
        if (ido == 1 || ido == -1)
            av_sym_laplacian_2d(nx, n, &workd[ipntr[0] - 1], &workd[ipntr[1] - 1]);
        else
            break;
    }
    int rvec = 1;
    dseupd_(&rvec, "A", sel.data(), d.data(), v.data(), &ldv, &sigma,
            "I", &n, "SM", &nev, &tol, resid.data(), &ncv,
            v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
    int nc = iparam[4];
    std::vector<Eig> out;
    for (int i = 0; i < nc; ++i) out.push_back({d[i], 0.0});
    return out;
}

static std::vector<Eig> f90_dndrv1(int nx) {
    int n = nx * nx, nev = 4, ncv = std::min(20, n - 1);
    int ldv = n, lworkl = 3 * ncv * ncv + 6 * ncv;
    double tol = 0.0, sigmar = 0.0, sigmai = 0.0;
    std::vector<double> v(ldv * ncv), workd(3 * n), workl(lworkl), resid(n);
    std::vector<double> dr(ncv + 1), di(ncv + 1), workev(3 * ncv);
    std::vector<int> sel(ncv);
    int iparam[11] = {}, ipntr[14] = {};
    iparam[0] = 1; iparam[2] = 300; iparam[6] = 1;
    int ido = 0, info = 0;
    while (true) {
        dnaupd_(&ido, "I", &n, "SM", &nev, &tol, resid.data(), &ncv,
                v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
        if (ido == 1 || ido == -1)
            av_conv_diff_2d(nx, n, 0.0, &workd[ipntr[0] - 1], &workd[ipntr[1] - 1]);
        else
            break;
    }
    int rvec = 1;
    dneupd_(&rvec, "A", sel.data(), dr.data(), di.data(), v.data(), &ldv,
            &sigmar, &sigmai, workev.data(),
            "I", &n, "SM", &nev, &tol, resid.data(), &ncv,
            v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
    int nc = iparam[4];
    std::vector<Eig> out;
    for (int i = 0; i < nc; ++i) out.push_back({dr[i], di[i]});
    return out;
}

static std::vector<Eig> f90_dsdrv2(int n) {
    int nev = 4, ncv = std::min(20, n - 1);
    int ldv = n, lworkl = ncv * (ncv + 8);
    double tol = 0.0, sigma = 0.0;
    double h2 = 1.0 / ((n + 1.0) * (n + 1.0));
    std::vector<double> ad(n, 2.0 / h2 - sigma), adl(n, -1.0 / h2),
                        adu(n, -1.0 / h2), adu2(n, 0.0);
    std::vector<int> ipiv(n);
    int ginfo = 0;
    dgttrf_(&n, adl.data(), ad.data(), adu.data(), adu2.data(), ipiv.data(), &ginfo);

    std::vector<double> v(ldv * ncv), workd(3 * n), workl(lworkl), resid(n), d(ncv * 2);
    std::vector<int> sel(ncv);
    int iparam[11] = {}, ipntr[11] = {};
    iparam[0] = 1; iparam[2] = 300; iparam[6] = 3;
    int ido = 0, info = 0, one = 1;
    while (true) {
        dsaupd_(&ido, "I", &n, "LM", &nev, &tol, resid.data(), &ncv,
                v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
        if (ido == 1 || ido == -1) {
            int inc = 1;
            dcopy_(&n, &workd[ipntr[0] - 1], &inc, &workd[ipntr[1] - 1], &inc);
            int sinfo = 0;
            dgttrs_("N", &n, &one, adl.data(), ad.data(), adu.data(), adu2.data(),
                    ipiv.data(), &workd[ipntr[1] - 1], &n, &sinfo);
        } else {
            break;
        }
    }
    int rvec = 1;
    dseupd_(&rvec, "A", sel.data(), d.data(), v.data(), &ldv, &sigma,
            "I", &n, "LM", &nev, &tol, resid.data(), &ncv,
            v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
    int nc = iparam[4];
    std::vector<Eig> out;
    for (int i = 0; i < nc; ++i) out.push_back({d[i], 0.0});
    return out;
}

static std::vector<Eig> f90_dndrv2(int n) {
    int nev = 4, ncv = std::min(20, n - 1);
    int ldv = n, lworkl = 3 * ncv * ncv + 6 * ncv;
    double tol = 0.0, sigmar = 1.0, sigmai = 0.0, rho = 10.0;
    double h = 1.0 / (n + 1), ss = rho * h / 2.0;
    std::vector<double> dd(n, 2.0 - sigmar), dl(n, -1.0 - ss),
                        du(n, -1.0 + ss), du2(n, 0.0);
    std::vector<int> ipiv(n);
    int ginfo = 0;
    dgttrf_(&n, dl.data(), dd.data(), du.data(), du2.data(), ipiv.data(), &ginfo);

    std::vector<double> v(ldv * ncv), workd(3 * n), workl(lworkl), resid(n);
    std::vector<double> dr(ncv + 1), di(ncv + 1), workev(3 * ncv);
    std::vector<int> sel(ncv);
    int iparam[11] = {}, ipntr[14] = {};
    iparam[0] = 1; iparam[2] = 300; iparam[6] = 3;
    int ido = 0, info = 0, one = 1;
    while (true) {
        dnaupd_(&ido, "I", &n, "LM", &nev, &tol, resid.data(), &ncv,
                v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
        if (ido == 1 || ido == -1) {
            int inc = 1;
            dcopy_(&n, &workd[ipntr[0] - 1], &inc, &workd[ipntr[1] - 1], &inc);
            int sinfo = 0;
            dgttrs_("N", &n, &one, dl.data(), dd.data(), du.data(), du2.data(),
                    ipiv.data(), &workd[ipntr[1] - 1], &n, &sinfo);
        } else {
            break;
        }
    }
    int rvec = 1;
    dneupd_(&rvec, "A", sel.data(), dr.data(), di.data(), v.data(), &ldv,
            &sigmar, &sigmai, workev.data(),
            "I", &n, "LM", &nev, &tol, resid.data(), &ncv,
            v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
    int nc = iparam[4];
    std::vector<Eig> out;
    for (int i = 0; i < nc; ++i) out.push_back({dr[i], di[i]});
    return out;
}

struct ProblemResult {
    const char* name;
    int n, nev;
    Timing cpp, f90;
    double max_rel_err;
    int nconv_cpp, nconv_f90;
};

static void print_table(const char* title, const std::vector<ProblemResult>& rows) {
    const int W = 92;
    std::string bar(W, '='), dash(W, '-');
    std::printf("\n%s\n %s\n%s\n", bar.c_str(), title, bar.c_str());
    std::printf(" %-10s %6s  %8s %8s %7s  %4s/%4s  %10s  %6s\n",
                "Problem", "n", "C++ us", "F90 us", "Ratio",
                "Nc++", "Nf90", "max|dEig|", "Status");
    std::printf(" %s\n", dash.c_str());

    for (auto& r : rows) {
        double ratio = r.f90.median_us > 0 ? double(r.f90.median_us) / r.cpp.median_us : 0.0;
        bool val_ok = r.max_rel_err < 1e-10
                      && r.nconv_cpp >= r.nev && r.nconv_f90 >= r.nev;
        const char* status = val_ok ? "OK" : "FAIL";
        std::printf(" %-10s %6d  %8lld %8lld %6.2fx  %4d/%4d  %10.2e  %6s\n",
                    r.name, r.n, r.cpp.median_us, r.f90.median_us, ratio,
                    r.nconv_cpp, r.nconv_f90, r.max_rel_err, status);
    }
}

static void print_summary(const std::vector<std::pair<const char*, std::vector<ProblemResult>>>& tiers) {
    const int W = 92;
    std::string bar(W, '=');
    std::printf("\n%s\n Summary: C++ / Fortran ratio (>1 = C++ faster)\n%s\n", bar.c_str(), bar.c_str());
    std::printf(" %-10s", "Problem");
    for (auto& [label, _] : tiers) std::printf(" %14s", label);
    std::printf("\n");
    std::printf(" %s\n", std::string(W, '-').c_str());

    const char* names[] = {"dsdrv1", "dndrv1", "dsdrv2", "dndrv2"};
    for (int p = 0; p < 4; ++p) {
        std::printf(" %-10s", names[p]);
        for (auto& [_, rows] : tiers) {
            for (auto& r : rows) {
                if (std::strcmp(r.name, names[p]) == 0) {
                    double ratio = r.f90.median_us > 0
                        ? double(r.f90.median_us) / r.cpp.median_us : 0.0;
                    std::printf(" %13.2fx", ratio);
                }
            }
        }
        std::printf("\n");
    }
    std::printf("%s\n", bar.c_str());
}

int main(int argc, char* argv[]) {
    int reps = 10, warmup = 3;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
    }

    struct Size { const char* label; int n; };
    Size sizes[] = {{"SMALL (n=100)", 100}, {"MEDIUM (n=2500)", 2500}, {"LARGE (n=10000)", 10000}};

    std::vector<std::pair<const char*, std::vector<ProblemResult>>> all_tiers;
    bool all_ok = true;

    for (auto& sz : sizes) {
        int nx = int(std::sqrt(double(sz.n)));
        int n2d = nx * nx;

        std::vector<ProblemResult> rows;

        {
            std::vector<Eig> ce, fe;
            auto ct = time_fn([&]{ ce = cpp_dsdrv1(nx); }, warmup, reps);
            auto ft = time_fn([&]{ fe = f90_dsdrv1(nx); }, warmup, reps);
            double err = eig_max_error(ce, fe);
            rows.push_back({"dsdrv1", n2d, 4, ct, ft, err, (int)ce.size(), (int)fe.size()});
        }
        {
            std::vector<Eig> ce, fe;
            auto ct = time_fn([&]{ ce = cpp_dndrv1(nx); }, warmup, reps);
            auto ft = time_fn([&]{ fe = f90_dndrv1(nx); }, warmup, reps);
            double err = eig_max_error(ce, fe);
            rows.push_back({"dndrv1", n2d, 4, ct, ft, err, (int)ce.size(), (int)fe.size()});
        }
        {
            std::vector<Eig> ce, fe;
            auto ct = time_fn([&]{ ce = cpp_dsdrv2(sz.n); }, warmup, reps);
            auto ft = time_fn([&]{ fe = f90_dsdrv2(sz.n); }, warmup, reps);
            double err = eig_max_error(ce, fe);
            rows.push_back({"dsdrv2", sz.n, 4, ct, ft, err, (int)ce.size(), (int)fe.size()});
        }
        {
            std::vector<Eig> ce, fe;
            auto ct = time_fn([&]{ ce = cpp_dndrv2(sz.n); }, warmup, reps);
            auto ft = time_fn([&]{ fe = f90_dndrv2(sz.n); }, warmup, reps);
            double err = eig_max_error(ce, fe);
            rows.push_back({"dndrv2", sz.n, 4, ct, ft, err, (int)ce.size(), (int)fe.size()});
        }

        for (auto& r : rows) {
            bool val_ok = r.max_rel_err < 1e-10
                          && r.nconv_cpp >= r.nev && r.nconv_f90 >= r.nev;
            if (!val_ok) all_ok = false;
        }

        print_table(sz.label, rows);
        all_tiers.push_back({sz.label, std::move(rows)});
    }

    print_summary(all_tiers);

    std::printf("\nOVERALL: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
