#ifndef ARNOLDI_DETAIL_HELPERS_HPP
#define ARNOLDI_DETAIL_HELPERS_HPP

// Small helper routines used by the callback-based Arnoldi solver:
//   - conv_check / nconv / sconv   — Ritz convergence checks
//   - ngets / sgets — shift selection

#include <arnoldi/detail/debug.hpp>
#include <arnoldi/detail/ops.hpp>
#include <arnoldi/detail/sort.hpp>
#include <arnoldi/detail/stats.hpp>

namespace arnoldi::detail {

  // Generic Ritz convergence check; mag(i) returns the magnitude of the
  // i-th Ritz value. Returns the count of converged values.
  template <typename Real, typename Magnitude>
  int conv_check(int n, Real* bounds, Real tol, Magnitude&& mag) {
    Real eps23 = Ops<Real>::lamch("Epsilon-Machine");
    eps23      = std::pow(eps23, Real(2.0 / 3.0));
    int nc     = 0;
    for (int i = 0; i < n; i++) {
      Real temp = std::max(eps23, mag(i));
      if (bounds[i] <= tol * temp) nc++;
    }
    return nc;
  }

  template <typename Real>
  void nconv(int n, Real* ritzr, Real* ritzi, Real* bounds, Real tol, int& nconv_out) {
    double t0, t1;
    arscnd(t0);
    nconv_out = conv_check<Real>(n, bounds, tol, [&](int i) { return Ops<Real>::lapy2(ritzr[i], ritzi[i]); });
    arscnd(t1);
    stats.conv += (t1 - t0);
  }

  template <typename Real>
  void sconv(int n, Real* ritz, Real* bounds, Real tol, int& nconv_out) {
    double t0, t1;
    arscnd(t0);
    nconv_out = conv_check<Real>(n, bounds, tol, [&](int i) { return std::abs(ritz[i]); });
    arscnd(t1);
    stats.conv += (t1 - t0);
  }

  // ngets — nonsymmetric shift selection.
  template <typename Real>
  void ngets(int ishift, const char* which, int& kev, int& np, Real* ritzr, Real* ritzi, Real* bounds, Real* shiftr,
             Real* shifti) {
    double t0, t1;
    arscnd(t0);
    int msglvl = debug.gets;

    if (std::strcmp(which, "LM") == 0)
      sortc<Real>("LR", true, kev + np, ritzr, ritzi, bounds);
    else if (std::strcmp(which, "SM") == 0)
      sortc<Real>("SR", true, kev + np, ritzr, ritzi, bounds);
    else if (std::strcmp(which, "LR") == 0)
      sortc<Real>("LM", true, kev + np, ritzr, ritzi, bounds);
    else if (std::strcmp(which, "SR") == 0)
      sortc<Real>("SM", true, kev + np, ritzr, ritzi, bounds);
    else if (std::strcmp(which, "LI") == 0)
      sortc<Real>("LM", true, kev + np, ritzr, ritzi, bounds);
    else if (std::strcmp(which, "SI") == 0)
      sortc<Real>("SM", true, kev + np, ritzr, ritzi, bounds);

    sortc<Real>(which, true, kev + np, ritzr, ritzi, bounds);

    if ((ritzr[np] - ritzr[np - 1]) == 0.0 && (ritzi[np] + ritzi[np - 1]) == 0.0) {
      np  = np - 1;
      kev = kev + 1;
    }

    if (ishift == 1) sortc<Real>("SR", true, np, bounds, ritzr, ritzi);

    arscnd(t1);
    stats.gets += (t1 - t0);

    if (msglvl > 0) {
      debug.ivout(1, &kev, "_ngets: KEV is");
      debug.ivout(1, &np, "_ngets: NP is");
      debug.vout(kev + np, ritzr, "_ngets: Eigenvalues of current H matrix -- real part");
      debug.vout(kev + np, ritzi, "_ngets: Eigenvalues of current H matrix -- imag part");
      debug.vout(kev + np, bounds, "_ngets: Ritz estimates of the current KEV+NP Ritz values");
    }
  }

  // sgets — symmetric shift selection.
  template <typename Real>
  void sgets(int ishift, const char* which, int kev, int np, Real* ritz, Real* bounds, Real* shifts) {
    double t0, t1;
    arscnd(t0);
    int msglvl = debug.gets;

    if (std::strcmp(which, "BE") == 0) {
      sortr<Real>("LA", true, kev + np, ritz, bounds);
      int kevd2 = kev / 2;
      if (kev > 1) {
        Ops<Real>::swap(std::min(kevd2, np), ritz, 1, &ritz[std::max(kevd2, np)], 1);
        Ops<Real>::swap(std::min(kevd2, np), bounds, 1, &bounds[std::max(kevd2, np)], 1);
      }
    } else {
      sortr<Real>(which, true, kev + np, ritz, bounds);
    }

    if (ishift == 1 && np > 0) {
      sortr<Real>("SM", true, np, bounds, ritz);
      Ops<Real>::copy(np, ritz, 1, shifts, 1);
    }

    arscnd(t1);
    stats.gets += (t1 - t0);

    if (msglvl > 0) {
      debug.ivout(1, &kev, "_sgets: KEV is");
      debug.ivout(1, &np, "_sgets: NP is");
      debug.vout(kev + np, ritz, "_sgets: Eigenvalues of current H matrix");
      debug.vout(kev + np, bounds, "_sgets: Associated Ritz estimates");
    }
  }

}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_HELPERS_HPP
