#ifndef ARNOLDI_DETAIL_EIG_HPP
#define ARNOLDI_DETAIL_EIG_HPP

// neigh — nonsymmetric Ritz values from upper Hessenberg matrix.
// seigt — symmetric Ritz values from tridiagonal matrix.
// stqrb — tridiagonal QR iteration (port of LAPACK dsteqr).

#include <arnoldi/detail/debug.hpp>
#include <arnoldi/detail/ops.hpp>
#include <arnoldi/detail/stats.hpp>

namespace arnoldi::detail {

  template <typename Real>
  void neigh(Real rnorm, int n, Real* h, int ldh, Real* ritzr, Real* ritzi, Real* bounds, Real* q, int ldq, Real* workl,
             int& ierr) {
    double t0, t1;
    arscnd(t0);
    int msglvl = debug.eigh;

    if (msglvl > 2) debug.mout(n, n, h, ldh, "_neigh: Entering upper Hessenberg matrix H ");

    Ops<Real>::lacpy("All", n, n, h, ldh, workl, n);
    for (int j = 0; j < n - 1; j++) bounds[j] = Real(0);
    bounds[n - 1] = Real(1);

    {
      int ilo = 1, ihi = n, wantt = 1, wantz = 1;
      Ops<Real>::lahqr_(&wantt, &wantz, &n, &ilo, &ihi, workl, &n, ritzr, ritzi, &ilo, &ilo, bounds, &ilo, &ierr);
    }

    if (ierr != 0) {
      arscnd(t1);
      stats.eigh += (t1 - t0);
      return;
    }

    if (msglvl > 1) debug.vout(n, bounds, "_neigh: last row of the Schur matrix for H");

    {
      Real vl = Real(0);
      int  nv = 0, select_dummy = 0;
      Ops<Real>::trevc_("R", "A", &select_dummy, &n, workl, &n, &vl, &n, q, &ldq, &n, &nv, &workl[n * n], &ierr);
    }

    if (ierr != 0) {
      arscnd(t1);
      stats.eigh += (t1 - t0);
      return;
    }

    int iconj = 0;
    for (int i = 0; i < n; i++) {
      if (std::abs(ritzi[i]) <= Real(0)) {
        Real temp = Ops<Real>::nrm2(n, &q[i * ldq], 1);
        Ops<Real>::scal(n, Real(1) / temp, &q[i * ldq], 1);
      } else if (iconj == 0) {
        Real temp = Ops<Real>::lapy2(Ops<Real>::nrm2(n, &q[i * ldq], 1), Ops<Real>::nrm2(n, &q[(i + 1) * ldq], 1));
        Ops<Real>::scal(n, Real(1) / temp, &q[i * ldq], 1);
        Ops<Real>::scal(n, Real(1) / temp, &q[(i + 1) * ldq], 1);
        iconj = 1;
      } else {
        iconj = 0;
      }
    }

    Ops<Real>::gemv("T", n, n, Real(1), q, ldq, bounds, 1, Real(0), workl, 1);

    if (msglvl > 1) debug.vout(n, workl, "_neigh: Last row of the eigenvector matrix for H");

    iconj = 0;
    for (int i = 0; i < n; i++) {
      if (std::abs(ritzi[i]) <= Real(0)) {
        bounds[i] = rnorm * std::abs(workl[i]);
      } else if (iconj == 0) {
        bounds[i]     = rnorm * Ops<Real>::lapy2(workl[i], workl[i + 1]);
        bounds[i + 1] = bounds[i];
        iconj         = 1;
      } else {
        iconj = 0;
      }
    }

    if (msglvl > 2) {
      debug.vout(n, ritzr, "_neigh: Real part of the eigenvalues of H");
      debug.vout(n, ritzi, "_neigh: Imaginary part of the eigenvalues of H");
      debug.vout(n, bounds, "_neigh: Ritz estimates for the eigenvalues of H");
    }

    arscnd(t1);
    stats.eigh += (t1 - t0);
  }

  template <typename Real>
  void stqrb(int n, Real* d, Real* e, Real* z, Real* work, int& info) {
    const Real zero = Real(0), one = Real(1), two = Real(2), three = Real(3);
    const int  maxit = 30;

    info             = 0;
    if (n == 0) return;
    if (n == 1) {
      z[0] = one;
      return;
    }

    Real eps    = Ops<Real>::lamch("e");
    Real eps2   = eps * eps;
    Real safmin = Ops<Real>::lamch("s");
    Real safmax = one / safmin;
    Real ssfmax = std::sqrt(safmax) / three;
    Real ssfmin = std::sqrt(safmin) / eps2;

    for (int j = 1; j <= n - 1; j++) z[j - 1] = zero;
    z[n - 1]   = one;

    int nmaxit = n * maxit;
    int jtot   = 0;
    int l1     = 1;
    int nm1    = n - 1;

  label_10:
    if (l1 > n) goto label_160;
    if (l1 > 1) e[l1 - 2] = zero;
    int m;
    if (l1 <= nm1) {
      for (m = l1; m <= nm1; m++) {
        Real tst = std::abs(e[m - 1]);
        if (tst == zero) goto label_30;
        if (tst <= std::sqrt(std::abs(d[m - 1])) * std::sqrt(std::abs(d[m])) * eps) {
          e[m - 1] = zero;
          goto label_30;
        }
      }
    }
    m = n;

  label_30 : {
    int l      = l1;
    int lsv    = l;
    int lend   = m;
    int lendsv = lend;
    l1         = m + 1;
    if (lend == l) goto label_10;

    Real anorm;
    {
      int tmp_n = lend - l + 1;
      anorm     = Ops<Real>::lanst_("i", &tmp_n, &d[l - 1], &e[l - 1]);
    }
    int iscale = 0;
    if (anorm == zero) goto label_10;
    if (anorm > ssfmax) {
      iscale     = 1;
      int i_zero = 0, i_one = 1;
      int m1 = lend - l + 1, m2 = lend - l;
      Ops<Real>::lascl_("g", &i_zero, &i_zero, &anorm, &ssfmax, &m1, &i_one, &d[l - 1], &n, &info);
      Ops<Real>::lascl_("g", &i_zero, &i_zero, &anorm, &ssfmax, &m2, &i_one, &e[l - 1], &n, &info);
    } else if (anorm < ssfmin) {
      iscale     = 2;
      int i_zero = 0, i_one = 1;
      int m1 = lend - l + 1, m2 = lend - l;
      Ops<Real>::lascl_("g", &i_zero, &i_zero, &anorm, &ssfmin, &m1, &i_one, &d[l - 1], &n, &info);
      Ops<Real>::lascl_("g", &i_zero, &i_zero, &anorm, &ssfmin, &m2, &i_one, &e[l - 1], &n, &info);
    }

    if (std::abs(d[lend - 1]) < std::abs(d[l - 1])) {
      lend = lsv;
      l    = lendsv;
    }

    if (lend > l) {
    label_40:
      if (l != lend) {
        int lendm1 = lend - 1;
        for (m = l; m <= lendm1; m++) {
          Real tst = std::pow(std::abs(e[m - 1]), 2);
          if (tst <= eps2 * std::abs(d[m - 1]) * std::abs(d[m]) + safmin) goto label_60;
        }
      }
      m = lend;
    label_60:
      if (m < lend) e[m - 1] = zero;
      Real p = d[l - 1];
      if (m == l) goto label_80;

      if (m == l + 1) {
        Real rt1, rt2, c, s;
        Ops<Real>::laev2_(&d[l - 1], &e[l - 1], &d[l], &rt1, &rt2, &c, &s);
        work[l - 1]     = c;
        work[n - 2 + l] = s;
        Real tst        = z[l];
        z[l]            = c * tst - s * z[l - 1];
        z[l - 1]        = s * tst + c * z[l - 1];
        d[l - 1]        = rt1;
        d[l]            = rt2;
        e[l - 1]        = zero;
        l += 2;
        if (l <= lend) goto label_40;
        goto label_140;
      }

      if (jtot == nmaxit) goto label_140;
      jtot++;

      {
        Real g = (d[l] - p) / (two * e[l - 1]);
        Real r = Ops<Real>::lapy2(g, one);
        g      = d[m - 1] - p + e[l - 1] / (g + std::copysign(r, g));
        Real s = one, c = one;
        p       = zero;
        int mm1 = m - 1;
        for (int i = mm1; i >= l; i--) {
          Real f = s * e[i - 1];
          Real b = c * e[i - 1];
          Ops<Real>::lartg_(&g, &f, &c, &s, &r);
          if (i != m - 1) e[i] = r;
          g               = d[i] - p;
          r               = (d[i - 1] - g) * s + two * c * b;
          p               = s * r;
          d[i]            = g + p;
          g               = c * r - b;
          work[i - 1]     = c;
          work[n - 2 + i] = -s;
        }
        {
          int mm    = m - l + 1;
          int i_one = 1;
          Ops<Real>::lasr_("r", "v", "b", &i_one, &mm, &work[l - 1], &work[n - 2 + l], &z[l - 1], &i_one);
        }
        d[l - 1] -= p;
        e[l - 1] = g;
      }
      goto label_40;

    label_80:
      d[l - 1] = p;
      l++;
      if (l <= lend) goto label_40;
      goto label_140;

    } else {
    label_90:
      if (l != lend) {
        int lendp1 = lend + 1;
        for (m = l; m >= lendp1; m--) {
          Real tst = std::pow(std::abs(e[m - 2]), 2);
          if (tst <= eps2 * std::abs(d[m - 1]) * std::abs(d[m - 2]) + safmin) goto label_110;
        }
      }
      m = lend;
    label_110:
      if (m > lend) e[m - 2] = zero;
      Real p = d[l - 1];
      if (m == l) goto label_130;

      if (m == l - 1) {
        Real rt1, rt2, c, s;
        Ops<Real>::laev2_(&d[l - 2], &e[l - 2], &d[l - 1], &rt1, &rt2, &c, &s);
        Real tst = z[l - 1];
        z[l - 1] = c * tst - s * z[l - 2];
        z[l - 2] = s * tst + c * z[l - 2];
        d[l - 2] = rt1;
        d[l - 1] = rt2;
        e[l - 2] = zero;
        l -= 2;
        if (l >= lend) goto label_90;
        goto label_140;
      }

      if (jtot == nmaxit) goto label_140;
      jtot++;

      {
        Real g = (d[l - 2] - p) / (two * e[l - 2]);
        Real r = Ops<Real>::lapy2(g, one);
        g      = d[m - 1] - p + e[l - 2] / (g + std::copysign(r, g));
        Real s = one, c = one;
        p       = zero;
        int lm1 = l - 1;
        for (int i = m; i <= lm1; i++) {
          Real f = s * e[i - 1];
          Real b = c * e[i - 1];
          Ops<Real>::lartg_(&g, &f, &c, &s, &r);
          if (i != m) e[i - 2] = r;
          g               = d[i - 1] - p;
          r               = (d[i] - g) * s + two * c * b;
          p               = s * r;
          d[i - 1]        = g + p;
          g               = c * r - b;
          work[i - 1]     = c;
          work[n - 2 + i] = s;
        }
        {
          int mm    = l - m + 1;
          int i_one = 1;
          Ops<Real>::lasr_("r", "v", "f", &i_one, &mm, &work[m - 1], &work[n - 2 + m], &z[m - 1], &i_one);
        }
        d[l - 1] -= p;
        e[lm1 - 1] = g;
      }
      goto label_90;

    label_130:
      d[l - 1] = p;
      l--;
      if (l >= lend) goto label_90;
      goto label_140;
    }

  label_140:
    if (iscale == 1) {
      int i_zero = 0, i_one = 1;
      int m1 = lendsv - lsv + 1, m2 = lendsv - lsv;
      Ops<Real>::lascl_("g", &i_zero, &i_zero, &ssfmax, &anorm, &m1, &i_one, &d[lsv - 1], &n, &info);
      Ops<Real>::lascl_("g", &i_zero, &i_zero, &ssfmax, &anorm, &m2, &i_one, &e[lsv - 1], &n, &info);
    } else if (iscale == 2) {
      int i_zero = 0, i_one = 1;
      int m1 = lendsv - lsv + 1, m2 = lendsv - lsv;
      Ops<Real>::lascl_("g", &i_zero, &i_zero, &ssfmin, &anorm, &m1, &i_one, &d[lsv - 1], &n, &info);
      Ops<Real>::lascl_("g", &i_zero, &i_zero, &ssfmin, &anorm, &m2, &i_one, &e[lsv - 1], &n, &info);
    }

    if (jtot < nmaxit) goto label_10;
    for (int i = 1; i <= n - 1; i++)
      if (e[i - 1] != zero) info++;
    goto label_190;
  }

  label_160:
    for (int ii = 2; ii <= n; ii++) {
      int  i = ii - 1;
      int  k = i;
      Real p = d[i - 1];
      for (int j = ii; j <= n; j++) {
        if (d[j - 1] < p) {
          k = j;
          p = d[j - 1];
        }
      }
      if (k != i) {
        d[k - 1] = d[i - 1];
        d[i - 1] = p;
        p        = z[k - 1];
        z[k - 1] = z[i - 1];
        z[i - 1] = p;
      }
    }

  label_190:
    return;
  }

  template <typename Real>
  void seigt(Real rnorm, int n, Real* h, int ldh, Real* eig, Real* bounds, Real* workl, int& ierr) {
    double t0, t1;
    arscnd(t0);
    int msglvl = debug.eigh;

    if (msglvl > 0) {
      debug.vout(n, &h[ldh], "_seigt: main diagonal of matrix H");
      if (n > 1) debug.vout(n - 1, &h[1], "_seigt: sub diagonal of matrix H");
    }

    Ops<Real>::copy(n, &h[ldh], 1, eig, 1);
    Ops<Real>::copy(n - 1, &h[1], 1, workl, 1);
    stqrb<Real>(n, eig, workl, bounds, &workl[n], ierr);
    if (ierr == 0) {
      if (msglvl > 1) debug.vout(n, bounds, "_seigt: last row of the eigenvector matrix for H");
      for (int k = 0; k < n; k++) bounds[k] = rnorm * std::abs(bounds[k]);
    }

    arscnd(t1);
    stats.eigh += (t1 - t0);
  }

}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_EIG_HPP
