#ifndef ARNOLDI_DETAIL_NONSYM_HPP
#define ARNOLDI_DETAIL_NONSYM_HPP

#include <algorithm>
#include <arnoldi/comm.hpp>
#include <arnoldi/detail/debug.hpp>
#include <arnoldi/detail/eig.hpp>
#include <arnoldi/detail/getv0.hpp>
#include <arnoldi/detail/helpers.hpp>
#include <arnoldi/detail/ops.hpp>
#include <arnoldi/detail/sort.hpp>
#include <arnoldi/detail/stats.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace arnoldi::detail {

  // naitr — Arnoldi iteration (nonsymmetric).
  template <typename Real, typename OP, typename BOP, typename Comm>
  void naitr(const char* bmat, int n, int k, int np, int nb, Real* resid, Real& rnorm, Real* v, int ldv, Real* h, int ldh,
             Real* workd, int& info, OP&& op, BOP&& bop, const Comm& comm) {
    const int ipj = 0, irj = n, ivj = 2 * n;

    Real      unfl = detail::Ops<Real>::lamch("safe minimum");
    Real      ovfl = (Real)1 / unfl;
    detail::Ops<Real>::labad(unfl, ovfl);
    Real   ulp    = detail::Ops<Real>::lamch("precision");
    Real   smlnum = unfl * ((Real)n / ulp);

    int    msglvl = detail::debug.aitr;
    double t0, t1, t2, t3, t4, t5;
    detail::arscnd(t0);
    info = 0;

    for (int j = k + 1; j <= k + np; ++j) {
      if (msglvl > 1) {
        int jj = j;
        detail::debug.ivout(1, &jj, "_naitr: generating Arnoldi vector number");
        detail::debug.vout(1, &rnorm, "_naitr: B-norm of the current residual is");
      }

      Real betaj  = rnorm;
      bool rstart = false;

      if (rnorm <= (Real)0) {
        if (msglvl > 0) {
          int jj = j;
          detail::debug.ivout(1, &jj, "_naitr: ****** RESTART AT STEP ******");
        }
        betaj = (Real)0;
        detail::stats.nrstrt++;
        rstart   = true;

        int ierr = -1;
        for (int itry = 1; itry <= 3; ++itry) {
          getv0<Real>(bmat, itry, false, n, j, v, ldv, resid, rnorm, workd, ierr, op, bop, comm);
          if (ierr >= 0) break;
        }
        if (ierr < 0) {
          info = j - 1;
          detail::arscnd(t1);
          detail::stats.aitr += (t1 - t0);
          return;
        }
      }

      detail::Ops<Real>::copy(n, resid, 1, &v[(j - 1) * ldv], 1);
      if (rnorm >= unfl) {
        Real temp1 = (Real)1 / rnorm;
        detail::Ops<Real>::scal(n, temp1, &v[(j - 1) * ldv], 1);
        detail::Ops<Real>::scal(n, temp1, &workd[ipj], 1);
      } else {
        int  ione = 1;
        int  infol;
        Real done = (Real)1;
        detail::Ops<Real>::lascl_("General", &ione, &ione, &rnorm, &done, &n, &ione, &v[(j - 1) * ldv], &n, &infol);
        detail::Ops<Real>::lascl_("General", &ione, &ione, &rnorm, &done, &n, &ione, &workd[ipj], &n, &infol);
      }

      detail::stats.nopx++;
      detail::arscnd(t2);
      detail::Ops<Real>::copy(n, &v[(j - 1) * ldv], 1, &workd[ivj], 1);
      op(&workd[ivj], &workd[irj]);
      detail::arscnd(t3);
      detail::stats.mvopx += (t3 - t2);

      detail::Ops<Real>::copy(n, &workd[irj], 1, resid, 1);

      // B-multiply w = OP*v_j (or copy through if bmat='I').
      detail::arscnd(t2);
      if (bmat[0] == 'G') {
        detail::stats.nbx++;
        bop(&workd[irj], &workd[ipj]);
      } else {
        detail::Ops<Real>::copy(n, resid, 1, &workd[ipj], 1);
      }
      if (bmat[0] == 'G') {
        detail::arscnd(t3);
        detail::stats.mvbx += (t3 - t2);
      }

      Real wnorm;
      if (bmat[0] == 'G') {
        wnorm = detail::pdot<Real>(comm, n, resid, 1, &workd[ipj], 1);
        wnorm = std::sqrt(std::abs(wnorm));
      } else {
        wnorm = detail::pnrm2_real<Real>(comm, n, resid, 1);
      }

      // h(:,j) = V' * B*w ;  resid -= V * h(:,j)
      detail::Ops<Real>::gemv("T", n, j, (Real)1, v, ldv, &workd[ipj], 1, (Real)0, &h[(j - 1) * ldh], 1);
      comm.allreduce_sum(&h[(j - 1) * ldh], j);
      detail::Ops<Real>::gemv("N", n, j, (Real)-1, v, ldv, &h[(j - 1) * ldh], 1, (Real)1, resid, 1);

      if (j > 1) h[(j - 2) * ldh + (j - 1)] = betaj;

      detail::arscnd(t4);

      detail::arscnd(t2);
      if (bmat[0] == 'G') {
        detail::stats.nbx++;
        detail::Ops<Real>::copy(n, resid, 1, &workd[irj], 1);
        bop(&workd[irj], &workd[ipj]);
      } else {
        detail::Ops<Real>::copy(n, resid, 1, &workd[ipj], 1);
      }
      if (bmat[0] == 'G') {
        detail::arscnd(t3);
        detail::stats.mvbx += (t3 - t2);
      }

      if (bmat[0] == 'G') {
        rnorm = detail::pdot<Real>(comm, n, resid, 1, &workd[ipj], 1);
        rnorm = std::sqrt(std::abs(rnorm));
      } else {
        rnorm = detail::pnrm2_real<Real>(comm, n, resid, 1);
      }

      if (rnorm <= 0.717 * wnorm) {
        detail::stats.nrorth++;
        for (int iter = 0; iter <= 1; ++iter) {
          if (msglvl > 2) {
            Real xtemp[2] = {wnorm, rnorm};
            detail::debug.vout(2, xtemp, "_naitr: re-orthonalization; wnorm and rnorm are");
            detail::debug.vout(j, &h[(j - 1) * ldh], "_naitr: j-th column of H");
          }

          detail::Ops<Real>::gemv("T", n, j, (Real)1, v, ldv, &workd[ipj], 1, (Real)0, &workd[irj], 1);
          comm.allreduce_sum(&workd[irj], j);
          detail::Ops<Real>::gemv("N", n, j, (Real)-1, v, ldv, &workd[irj], 1, (Real)1, resid, 1);
          detail::Ops<Real>::axpy(j, (Real)1, &workd[irj], 1, &h[(j - 1) * ldh], 1);

          detail::arscnd(t2);
          Real rnorm1;
          if (bmat[0] == 'G') {
            detail::stats.nbx++;
            detail::Ops<Real>::copy(n, resid, 1, &workd[irj], 1);
            bop(&workd[irj], &workd[ipj]);
          } else {
            detail::Ops<Real>::copy(n, resid, 1, &workd[ipj], 1);
          }
          if (bmat[0] == 'G') {
            detail::arscnd(t3);
            detail::stats.mvbx += (t3 - t2);
          }

          if (bmat[0] == 'G') {
            rnorm1 = detail::pdot<Real>(comm, n, resid, 1, &workd[ipj], 1);
            rnorm1 = std::sqrt(std::abs(rnorm1));
          } else {
            rnorm1 = detail::pnrm2_real<Real>(comm, n, resid, 1);
          }

          if (msglvl > 0 && iter > 0) {
            int jj = j;
            detail::debug.ivout(1, &jj, "_naitr: Iterative refinement for Arnoldi residual");
            if (msglvl > 2) {
              Real xtemp2[2] = {rnorm, rnorm1};
              detail::debug.vout(2, xtemp2, "_naitr: iterative refinement ; rnorm and rnorm1 are");
            }
          }

          if (rnorm1 > 0.717 * rnorm) {
            rnorm = rnorm1;
            break;
          }
          detail::stats.nitref++;
          rnorm = rnorm1;
          if (iter == 1) {
            for (int jj = 0; jj < n; jj++) resid[jj] = (Real)0;
            rnorm = (Real)0;
          }
        }
      }

      rstart = false;
      detail::arscnd(t5);
      detail::stats.itref += (t5 - t4);
    }

    // Zero small sub-diagonal entries.
    detail::arscnd(t1);
    detail::stats.aitr += (t1 - t0);
    for (int i = std::max(1, k); i <= k + np - 1; i++) {
      Real tst1 = std::abs(h[(i - 1) * ldh + (i - 1)]) + std::abs(h[i * ldh + i]);
      if (tst1 == (Real)0) {
        int kknp = k + np;
        tst1     = detail::Ops<Real>::lanhs("1", kknp, h, ldh, &workd[n]);
      }
      if (std::abs(h[(i - 1) * ldh + i]) <= std::max(ulp * tst1, smlnum)) h[(i - 1) * ldh + i] = (Real)0;
    }

    if (msglvl > 2) {
      int knp = k + np;
      detail::debug.mout(knp, knp, h, ldh, "_naitr: Final upper Hessenberg matrix H of order K+NP");
    }
  }

  // napps — apply implicit shifts to nonsymmetric Arnoldi factorisation.
  // Ported from ARPACK dnapps.f (goto-free version).
  template <typename Real>
  void napps(int n, int& kev, int np, Real* shiftr, Real* shifti, Real* v, int ldv, Real* h, int ldh, Real* resid, Real* q,
             int ldq, Real* workl, Real* workd) {
    Real unfl = detail::Ops<Real>::lamch("safe minimum");
    Real ovfl = Real(1) / unfl;
    detail::Ops<Real>::labad(unfl, ovfl);
    Real   ulp    = detail::Ops<Real>::lamch("precision");
    Real   smlnum = unfl * (Real(n) / ulp);

    double t0, t1;
    detail::arscnd(t0);
    int msglvl = detail::debug.apps;
    int kplusp = kev + np;

    {
      Real fzero = Real(0), fone = Real(1);
      detail::Ops<Real>::laset_("All", &kplusp, &kplusp, &fzero, &fone, q, &ldq);
    }

    if (np == 0) {
      detail::arscnd(t1);
      detail::stats.apps += (t1 - t0);
      return;
    }

    bool cconj = false;

    for (int jj = 1; jj <= np; jj++) {
      Real sigmar = shiftr[jj - 1];
      Real sigmai = shifti[jj - 1];

      if (msglvl > 2) {
        detail::debug.ivout(1, &jj, "_napps: shift number.");
        detail::debug.vout(1, &sigmar, "_napps: The real part of the shift ");
        detail::debug.vout(1, &sigmai, "_napps: The imaginary part of the shift ");
      }

      if (cconj) {
        cconj = false;
        continue;
      }
      if (jj < np && std::abs(sigmai) > Real(0))
        cconj = true;
      else if (jj == np && std::abs(sigmai) > Real(0)) {
        kev++;
        continue;
      }

      int istart = 1;

      while (istart <= kplusp) {
        int iend = kplusp;
        for (int i = istart; i <= kplusp - 1; i++) {
          Real tst1 = std::abs(h[(i - 1) * ldh + (i - 1)]) + std::abs(h[i * ldh + i]);
          if (tst1 == Real(0)) {
            int kk = kplusp - jj + 1;
            tst1   = detail::Ops<Real>::lanhs("1", kk, h, ldh, workl);
          }
          if (std::abs(h[(i - 1) * ldh + i]) <= std::max(ulp * tst1, smlnum)) {
            if (msglvl > 0) {
              detail::debug.ivout(1, &i, "_napps: matrix splitting at row/column no.");
              detail::debug.ivout(1, &jj, "_napps: matrix splitting with shift number.");
              detail::debug.vout(1, &h[(i - 1) * ldh + i], "_napps: off diagonal element.");
            }
            iend                 = i;
            h[(i - 1) * ldh + i] = Real(0);
            break;
          }
        }

        if (istart != iend && !(istart + 1 == iend && std::abs(sigmai) > Real(0))) {
          if (msglvl > 2) {
            detail::debug.ivout(1, &istart, "_napps: Start of current block ");
            detail::debug.ivout(1, &iend, "_napps: End of current block ");
          }

          Real h11 = h[(istart - 1) * ldh + (istart - 1)];
          Real h21 = h[(istart - 1) * ldh + istart];

          if (std::abs(sigmai) <= Real(0)) {
            Real f = h11 - sigmar;
            Real g = h21;

            for (int i = istart; i <= iend - 1; i++) {
              Real c, s, r;
              detail::Ops<Real>::lartg_(&f, &g, &c, &s, &r);
              if (i > istart) {
                if (r < Real(0)) {
                  r = -r;
                  c = -c;
                  s = -s;
                }
                h[(i - 2) * ldh + (i - 1)] = r;
                h[(i - 2) * ldh + i]       = Real(0);
              }
              for (int j = i; j <= kplusp; j++) {
                Real t                     = c * h[(j - 1) * ldh + (i - 1)] + s * h[(j - 1) * ldh + i];
                h[(j - 1) * ldh + i]       = -s * h[(j - 1) * ldh + (i - 1)] + c * h[(j - 1) * ldh + i];
                h[(j - 1) * ldh + (i - 1)] = t;
              }
              for (int j = 1; j <= std::min(i + 2, iend); j++) {
                Real t                     = c * h[(i - 1) * ldh + (j - 1)] + s * h[i * ldh + (j - 1)];
                h[i * ldh + (j - 1)]       = -s * h[(i - 1) * ldh + (j - 1)] + c * h[i * ldh + (j - 1)];
                h[(i - 1) * ldh + (j - 1)] = t;
              }
              for (int j = 1; j <= std::min(i + jj, kplusp); j++) {
                Real t                     = c * q[(i - 1) * ldq + (j - 1)] + s * q[i * ldq + (j - 1)];
                q[i * ldq + (j - 1)]       = -s * q[(i - 1) * ldq + (j - 1)] + c * q[i * ldq + (j - 1)];
                q[(i - 1) * ldq + (j - 1)] = t;
              }
              if (i < iend - 1) {
                f = h[(i - 1) * ldh + i];
                g = h[(i - 1) * ldh + (i + 1)];
              }
            }
          } else {
            Real h12 = h[istart * ldh + (istart - 1)];
            Real h22 = h[istart * ldh + istart];
            Real h32 = h[istart * ldh + (istart + 1)];

            Real s2  = Real(2) * sigmar;
            Real t2  = detail::Ops<Real>::lapy2(sigmar, sigmai);
            Real u[3];
            u[0] = (h11 * (h11 - s2) + t2 * t2) / h21 + h12;
            u[1] = h11 + h22 - s2;
            u[2] = h32;

            for (int i = istart; i <= iend - 1; i++) {
              int  nr = std::min(3, iend - i + 1);
              Real tau;
              int  ione = 1;
              detail::Ops<Real>::larfg_(&nr, &u[0], &u[1], &ione, &tau);
              if (i > istart) {
                h[(i - 2) * ldh + (i - 1)] = u[0];
                h[(i - 2) * ldh + i]       = Real(0);
                if (i < iend - 1) h[(i - 2) * ldh + (i + 1)] = Real(0);
              }
              u[0]      = Real(1);
              int ncols = kplusp - i + 1;
              detail::Ops<Real>::larf_("Left", &nr, &ncols, u, &ione, &tau, &h[(i - 1) * ldh + (i - 1)], &ldh, workl);
              int ir = std::min(i + 3, iend);
              detail::Ops<Real>::larf_("Right", &ir, &nr, u, &ione, &tau, &h[(i - 1) * ldh], &ldh, workl);
              detail::Ops<Real>::larf_("Right", &kplusp, &nr, u, &ione, &tau, &q[(i - 1) * ldq], &ldq, workl);
              if (i < iend - 1) {
                u[0] = h[(i - 1) * ldh + i];
                u[1] = h[(i - 1) * ldh + (i + 1)];
                if (i < iend - 2) u[2] = h[(i - 1) * ldh + (i + 2)];
              }
            }
          }
        }
        istart = iend + 1;
      }
    }

    for (int j = 1; j <= kev; j++) {
      if (h[(j - 1) * ldh + j] < Real(0)) {
        int len1 = kplusp - j + 1;
        detail::Ops<Real>::scal(len1, Real(-1), &h[(j - 1) * ldh + j], ldh);
        int len2 = std::min(j + 2, kplusp);
        detail::Ops<Real>::scal(len2, Real(-1), &h[j * ldh], 1);
        int len3 = std::min(j + np + 1, kplusp);
        detail::Ops<Real>::scal(len3, Real(-1), &q[j * ldq], 1);
      }
    }

    for (int i = 1; i <= kev; i++) {
      Real tst1 = std::abs(h[(i - 1) * ldh + (i - 1)]) + std::abs(h[i * ldh + i]);
      if (tst1 == Real(0)) tst1 = detail::Ops<Real>::lanhs("1", kev, h, ldh, workl);
      if (h[(i - 1) * ldh + i] <= std::max(ulp * tst1, smlnum)) h[(i - 1) * ldh + i] = Real(0);
    }

    if (h[(kev - 1) * ldh + kev] > Real(0))
      detail::Ops<Real>::gemv("N", n, kplusp, Real(1), v, ldv, &q[kev * ldq], 1, Real(0), &workd[n], 1);

    for (int i = 1; i <= kev; i++) {
      detail::Ops<Real>::gemv("N", n, kplusp - i + 1, Real(1), v, ldv, &q[(kev - i) * ldq], 1, Real(0), workd, 1);
      detail::Ops<Real>::copy(n, workd, 1, &v[(kplusp - i) * ldv], 1);
    }

    detail::Ops<Real>::lacpy("A", n, kev, &v[(kplusp - kev) * ldv], ldv, v, ldv);

    if (h[(kev - 1) * ldh + kev] > Real(0)) detail::Ops<Real>::copy(n, &workd[n], 1, &v[kev * ldv], 1);

    detail::Ops<Real>::scal(n, q[(kev - 1) * ldq + (kplusp - 1)], resid, 1);
    if (h[(kev - 1) * ldh + kev] > Real(0)) detail::Ops<Real>::axpy(n, h[(kev - 1) * ldh + kev], &v[kev * ldv], 1, resid, 1);

    if (msglvl > 1) {
      detail::debug.vout(1, &q[(kev - 1) * ldq + (kplusp - 1)], "_napps: sigmak = (e_{kev+p}^T*Q)*e_{kev}");
      detail::debug.vout(1, &h[(kev - 1) * ldh + kev], "_napps: betak = e_{kev+1}^T*H*e_{kev}");
      detail::debug.ivout(1, &kev, "_napps: Order of the final Hessenberg matrix ");
      if (msglvl > 2) detail::debug.mout(kev, kev, h, ldh, "_napps: updated Hessenberg matrix H for next iteration");
    }

    detail::arscnd(t1);
    detail::stats.apps += (t1 - t0);
  }

  // naup2 — main Arnoldi iteration driver (nonsymmetric).
  template <typename Real, typename OP, typename BOP, typename Comm>
  void naup2(const char* bmat, int n, const char* which, int& nev, int& np, Real tol, Real* resid, int mode, int iupd, int ishift,
             int& mxiter, Real* v, int ldv, Real* h, int ldh, Real* ritzr, Real* ritzi, Real* bounds, Real* q, int ldq,
             Real* workl, Real* workd, int& info, OP&& op, BOP&& bop, const Comm& comm) {
    double t0, t1, t2, t3;
    detail::arscnd(t0);
    int  msglvl = detail::debug.aup2;

    Real eps23  = detail::Ops<Real>::lamch("Epsilon-Machine");
    eps23       = std::pow(eps23, (Real)(2.0 / 3.0));

    int  nev0   = nev;
    int  np0    = np;
    int  kplusp = nev + np;
    int  nconv  = 0;
    int  iter   = 0;
    Real rnorm  = 0;
    int  ierr;
    char wprime[3];

    bool initv = (info != 0);
    if (initv) info = 0;

    const auto end_naup2 = [&]() {
      detail::arscnd(t1);
      detail::stats.aup2 = t1 - t0;
    };

    getv0<Real>(bmat, 1, initv, n, 1, v, ldv, resid, rnorm, workd, info, op, bop, comm);
    if (rnorm == (Real)0) {
      info = -9;
      end_naup2();
      return;
    }

    naitr<Real>(bmat, n, 0, nev, mode, resid, rnorm, v, ldv, h, ldh, workd, info, op, bop, comm);
    if (info > 0) {
      np     = info;
      mxiter = iter;
      info   = -9999;
      end_naup2();
      return;
    }

    for (iter = 1; iter <= mxiter; ++iter) {
      if (msglvl > 0) {
        int it = iter;
        detail::debug.ivout(1, &it, "_naup2: **** Start of major iteration number ****");
      }

      np = kplusp - nev;

      if (msglvl > 1) {
        detail::debug.ivout(1, &nev, "_naup2: The length of the current Arnoldi factorization");
        detail::debug.ivout(1, &np, "_naup2: Extend the Arnoldi factorization by");
      }

      naitr<Real>(bmat, n, nev, np, mode, resid, rnorm, v, ldv, h, ldh, workd, info, op, bop, comm);
      if (info > 0) {
        np     = info;
        mxiter = iter;
        info   = -9999;
        end_naup2();
        return;
      }

      if (msglvl > 1) {
        detail::debug.vout(1, &rnorm, "_naup2: Corresponding B-norm of the residual");
      }

      detail::neigh<Real>(rnorm, kplusp, h, ldh, ritzr, ritzi, bounds, q, ldq, workl, ierr);
      if (ierr != 0) {
        info = -8;
        end_naup2();
        return;
      }

      detail::Ops<Real>::copy(kplusp, ritzr, 1, &workl[kplusp * kplusp], 1);
      detail::Ops<Real>::copy(kplusp, ritzi, 1, &workl[kplusp * kplusp + kplusp], 1);
      detail::Ops<Real>::copy(kplusp, bounds, 1, &workl[kplusp * kplusp + 2 * kplusp], 1);

      nev        = nev0;
      np         = np0;
      int numcnv = nev;
      detail::ngets<Real>(ishift, which, nev, np, ritzr, ritzi, bounds, workl, &workl[np]);
      if (nev == nev0 + 1) numcnv = nev0 + 1;

      detail::Ops<Real>::copy(nev, &bounds[np], 1, &workl[2 * np], 1);
      detail::nconv<Real>(nev, &ritzr[np], &ritzi[np], &workl[2 * np], tol, nconv);

      if (msglvl > 2) {
        int kp[4] = {nev, np, numcnv, nconv};
        detail::debug.ivout(4, kp, "_naup2: NEV, NP, NUMCNV, NCONV are");
        detail::debug.vout(kplusp, ritzr, "_naup2: Real part of the eigenvalues of H");
        detail::debug.vout(kplusp, ritzi, "_naup2: Imaginary part of the eigenvalues of H");
        detail::debug.vout(kplusp, bounds, "_naup2: Ritz estimates of the current NCV Ritz values");
      }

      {
        int nptemp = np;
        for (int j = 0; j < nptemp; j++) {
          if (bounds[j] == (Real)0) {
            np--;
            nev++;
          }
        }
      }

      if (nconv >= numcnv || iter > mxiter || np == 0) {
        if (msglvl > 4) {
          detail::debug.vout(kplusp, &workl[kplusp * kplusp], "_naup2: Real part of the eig computed by _neigh:");
          detail::debug.vout(kplusp, &workl[kplusp * kplusp + kplusp], "_naup2: Imag part of the eig computed by _neigh:");
          detail::debug.vout(kplusp, &workl[kplusp * kplusp + kplusp * 2], "_naup2: Ritz estimates computed by _neigh:");
        }

        h[2] = rnorm;

        if (std::strcmp(which, "LM") == 0) std::strcpy(wprime, "SR");
        if (std::strcmp(which, "SM") == 0) std::strcpy(wprime, "LR");
        if (std::strcmp(which, "LR") == 0) std::strcpy(wprime, "SM");
        if (std::strcmp(which, "SR") == 0) std::strcpy(wprime, "LM");
        if (std::strcmp(which, "LI") == 0) std::strcpy(wprime, "SM");
        if (std::strcmp(which, "SI") == 0) std::strcpy(wprime, "LM");
        detail::sortc<Real>(wprime, true, kplusp, ritzr, ritzi, bounds);

        if (std::strcmp(which, "LM") == 0) std::strcpy(wprime, "SM");
        if (std::strcmp(which, "SM") == 0) std::strcpy(wprime, "LM");
        if (std::strcmp(which, "LR") == 0) std::strcpy(wprime, "SR");
        if (std::strcmp(which, "SR") == 0) std::strcpy(wprime, "LR");
        if (std::strcmp(which, "LI") == 0) std::strcpy(wprime, "SI");
        if (std::strcmp(which, "SI") == 0) std::strcpy(wprime, "LI");
        detail::sortc<Real>(wprime, true, kplusp, ritzr, ritzi, bounds);

        for (int j = 0; j < numcnv; j++) {
          Real temp = std::max(eps23, detail::Ops<Real>::lapy2(ritzr[j], ritzi[j]));
          bounds[j] = bounds[j] / temp;
        }
        std::strcpy(wprime, "LR");
        detail::sortc<Real>(wprime, true, numcnv, bounds, ritzr, ritzi);
        for (int j = 0; j < numcnv; j++) {
          Real temp = std::max(eps23, detail::Ops<Real>::lapy2(ritzr[j], ritzi[j]));
          bounds[j] = bounds[j] * temp;
        }
        detail::sortc<Real>(which, true, nconv, ritzr, ritzi, bounds);

        if (msglvl > 1) {
          detail::debug.vout(kplusp, ritzr, "_naup2: Sorted real part of the eigenvalues");
          detail::debug.vout(kplusp, ritzi, "_naup2: Sorted imaginary part of the eigenvalues");
          detail::debug.vout(kplusp, bounds, "_naup2: Sorted ritz estimates.");
        }

        if (iter > mxiter && nconv < numcnv) info = 1;
        if (np == 0 && nconv < numcnv) info = 2;

        np     = nconv;
        mxiter = iter;
        nev    = nconv;
        end_naup2();
        return;

      } else if (nconv < numcnv && ishift == 1) {
        int nevbef = nev;
        nev        = nev + std::min(nconv, np / 2);
        if (nev == 1 && kplusp >= 6)
          nev = kplusp / 2;
        else if (nev == 1 && kplusp > 3)
          nev = 2;
        if (nev > kplusp - 2) nev = kplusp - 2;
        np = kplusp - nev;

        if (nevbef < nev) detail::ngets<Real>(ishift, which, nev, np, ritzr, ritzi, bounds, workl, &workl[np]);
      }

      if (msglvl > 0) {
        detail::debug.ivout(1, &nconv, "_naup2: no. of \"converged\" Ritz values at this iter.");
        if (msglvl > 1) {
          int kp[2] = {nev, np};
          detail::debug.ivout(2, kp, "_naup2: NEV and NP are");
          detail::debug.vout(nev, &ritzr[np], "_naup2: \"wanted\" Ritz values -- real part");
          detail::debug.vout(nev, &ritzi[np], "_naup2: \"wanted\" Ritz values -- imag part");
          detail::debug.vout(nev, &bounds[np], "_naup2: Ritz estimates of the \"wanted\" values ");
        }
      }

      napps<Real>(n, nev, np, ritzr, ritzi, v, ldv, h, ldh, resid, q, ldq, workl, workd);

      // B-norm of residual after compression.
      detail::arscnd(t2);
      if (bmat[0] == 'G') {
        detail::stats.nbx++;
        detail::Ops<Real>::copy(n, resid, 1, &workd[n], 1);
        bop(&workd[n], workd);
        rnorm = detail::pdot<Real>(comm, n, resid, 1, workd, 1);
        rnorm = std::sqrt(std::abs(rnorm));
      } else {
        detail::Ops<Real>::copy(n, resid, 1, workd, 1);
        rnorm = detail::pnrm2_real<Real>(comm, n, resid, 1);
      }
      if (bmat[0] == 'G') {
        detail::arscnd(t3);
        detail::stats.mvbx += (t3 - t2);
      }

      if (msglvl > 2) {
        detail::debug.vout(1, &rnorm, "_naup2: B-norm of residual for compressed factorization");
        detail::debug.mout(nev, nev, h, ldh, "_naup2: Compressed upper Hessenberg matrix H");
      }
    }

    // Exceeded mxiter.
    info = 1;
    end_naup2();
  }

  // naupd — nonsymmetric eigensolver entry point (callback version).
  template <typename Real, typename OP, typename BOP, typename Comm>
  void naupd(const char* bmat, int n, const char* which, int nev, Real& tol, Real* resid, int ncv, Real* v, int ldv, int* iparam,
             int* ipntr, Real* workd, Real* workl, int lworkl, int& info, OP&& op, BOP&& bop, const Comm& comm) {
    detail::stats.reset();
    double t0, t1;
    detail::arscnd(t0);
    int msglvl = detail::debug.aupd;

    int ishift = iparam[0];
    int mxiter = iparam[2];
    int nb     = 1;
    int iupd   = 1;
    int mode   = iparam[6];
    int ierr   = 0;

    if (n <= 0)
      ierr = -1;
    else if (nev <= 0)
      ierr = -2;
    else if (ncv <= nev + 1)
      ierr = -3;
    else if (mxiter <= 0)
      ierr = -4;
    else if (std::strcmp(which, "LM") != 0 && std::strcmp(which, "SM") != 0 && std::strcmp(which, "LR") != 0 &&
             std::strcmp(which, "SR") != 0 && std::strcmp(which, "LI") != 0 && std::strcmp(which, "SI") != 0)
      ierr = -5;
    else if (bmat[0] != 'I' && bmat[0] != 'G')
      ierr = -6;
    else if (lworkl < 3 * ncv * ncv + 6 * ncv)
      ierr = -7;
    else if (mode < 1 || mode > 4)
      ierr = -10;
    else if (mode == 1 && bmat[0] == 'G')
      ierr = -11;
    else if (ishift < 0 || ishift > 1)
      ierr = -12;

    if (ierr != 0) {
      info = ierr;
      return;
    }

    if (nb <= 0) nb = 1;
    if (tol <= (Real)0) tol = detail::Ops<Real>::lamch("EpsMach");

    int np   = ncv - nev;
    int nev0 = nev;

    for (int j = 0; j < 3 * ncv * ncv + 6 * ncv; j++) workl[j] = (Real)0;

    int ldh    = ncv;
    int ldq    = ncv;
    int ih     = 0;
    int ritzr  = ih + ldh * ncv;
    int ritzi  = ritzr + ncv;
    int bounds = ritzi + ncv;
    int iq     = bounds + ncv;
    int iw     = iq + ldq * ncv;
    int next   = iw + ncv * ncv + 3 * ncv;

    ipntr[3]   = next;
    ipntr[4]   = ih;
    ipntr[5]   = ritzr;
    ipntr[6]   = ritzi;
    ipntr[7]   = bounds;
    ipntr[13]  = iw;

    naup2<Real>(bmat, n, which, nev0, np, tol, resid, mode, iupd, ishift, mxiter, v, ldv, &workl[ih], ldh, &workl[ritzr],
                &workl[ritzi], &workl[bounds], &workl[iq], ldq, &workl[iw], workd, info, op, bop, comm);

    iparam[2]  = mxiter;
    iparam[4]  = np;
    iparam[8]  = detail::stats.nopx;
    iparam[9]  = detail::stats.nbx;
    iparam[10] = detail::stats.nrorth;

    if (info < 0) { /* already set */
    } else if (info == 2)
      info = 3;

    if (msglvl > 0) {
      std::printf(
          "\n"
          "     =============================================\n"
          "     = Nonsymmetric implicit Arnoldi update code =\n"
          "     = Version Number:  2.4  (callback)         =\n"
          "     =============================================\n"
          "     Total number update iterations             = %5d\n"
          "     Total number of OP*x operations            = %5d\n"
          "     Total number of B*x operations             = %5d\n"
          "     Total number of reorthogonalization steps  = %5d\n"
          "     Total number of iterative refinement steps = %5d\n"
          "     Total number of restart steps              = %5d\n"
          "     Total time in user OP*x operation          = %12.6f\n"
          "     Total time in user B*x operation           = %12.6f\n"
          "     Total time in Arnoldi update routine       = %12.6f\n"
          "     Total time in naup2 routine                = %12.6f\n"
          "     Total time in basic Arnoldi iteration loop = %12.6f\n"
          "     Total time in reorthogonalization phase    = %12.6f\n"
          "     Total time in (re)start vector generation  = %12.6f\n"
          "     Total time in Hessenberg eig. subproblem   = %12.6f\n"
          "     Total time in getting the shifts           = %12.6f\n"
          "     Total time in applying the shifts          = %12.6f\n"
          "     Total time in convergence testing          = %12.6f\n"
          "     Total time in computing final Ritz vectors = %12.6f\n",
          mxiter, detail::stats.nopx, detail::stats.nbx, detail::stats.nrorth, detail::stats.nitref, detail::stats.nrstrt,
          (Real)detail::stats.mvopx, (Real)detail::stats.mvbx, (Real)detail::stats.aupd, (Real)detail::stats.aup2,
          (Real)detail::stats.aitr, (Real)detail::stats.itref, (Real)detail::stats.getv0, (Real)detail::stats.eigh,
          (Real)detail::stats.gets, (Real)detail::stats.apps, (Real)detail::stats.conv, (Real)detail::stats.rvec);
    }

    detail::arscnd(t1);
    detail::stats.aupd = t1 - t0;
  }

  // op + bop, no comm (defaults to SerialComm).
  template <typename Real, typename OP, typename BOP>
  void naupd(const char* bmat, int n, const char* which, int nev, Real& tol, Real* resid, int ncv, Real* v, int ldv, int* iparam,
             int* ipntr, Real* workd, Real* workl, int lworkl, int& info, OP&& op, BOP&& bop) {
    naupd<Real>(bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, std::forward<OP>(op),
                std::forward<BOP>(bop), SerialComm{});
  }

  // Standard problem (bmat='I'), no bop, no comm.
  template <typename Real, typename OP>
  void naupd(const char* bmat, int n, const char* which, int nev, Real& tol, Real* resid, int ncv, Real* v, int ldv, int* iparam,
             int* ipntr, Real* workd, Real* workl, int lworkl, int& info, OP&& op) {
    naupd<Real>(
        bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, std::forward<OP>(op),
        [](const Real*, Real*) {}, SerialComm{});
  }

  // neupd — nonsymmetric eigenvector extraction (callback version).
  //
  // Reads the Hessenberg factorization and Ritz data left by cb::naupd in
  // workl/ipntr/iparam (cb::naupd stores 0-based offsets in ipntr[4..7,13])
  // and computes eigenvalues (dr,di) and optionally eigenvectors z.
  template <typename Real>
  void neupd(bool rvec, const char* howmny, Real* dr, Real* di, Real* z, int ldz, Real sigmar, Real sigmai, Real* workev,
             const char* bmat, int n, const char* which, int nev, Real tol, Real* resid, int ncv, Real* v, int ldv, int* iparam,
             int* ipntr, Real* workd, Real* workl, int lworkl, int& info) {
    int msglvl = detail::debug.eupd;
    int mode   = iparam[6];
    int nconv  = iparam[4];
    info       = 0;

    Real eps23 = detail::Ops<Real>::lamch("Epsilon-Machine");
    eps23      = std::pow(eps23, Real(2.0 / 3.0));

    int ierr   = 0;

    if (nconv <= 0)
      ierr = -14;
    else if (n <= 0)
      ierr = -1;
    else if (nev <= 0)
      ierr = -2;
    else if (ncv <= nev + 1)
      ierr = -3;
    else if (std::strcmp(which, "LM") != 0 && std::strcmp(which, "SM") != 0 && std::strcmp(which, "LR") != 0 &&
             std::strcmp(which, "SR") != 0 && std::strcmp(which, "LI") != 0 && std::strcmp(which, "SI") != 0)
      ierr = -5;
    else if (bmat[0] != 'I' && bmat[0] != 'G')
      ierr = -6;
    else if (lworkl < 3 * ncv * ncv + 6 * ncv)
      ierr = -7;
    else if (rvec && howmny[0] != 'A' && howmny[0] != 'P')
      ierr = -13;
    else if (rvec && howmny[0] == 'S')
      ierr = -12;

    char type[7];
    if (mode == 1 || mode == 2)
      std::strcpy(type, "REGULR");
    else if (mode == 3 && sigmai == Real(0))
      std::strcpy(type, "SHIFTI");
    else if (mode == 3)
      std::strcpy(type, "REALPT");
    else if (mode == 4)
      std::strcpy(type, "IMAGPT");
    else
      ierr = -10;
    if (mode == 1 && bmat[0] == 'G') ierr = -11;

    if (ierr != 0) {
      info = ierr;
      return;
    }

    int ih        = ipntr[4];
    int ritzr_    = ipntr[5];
    int ritzi_    = ipntr[6];
    int bounds    = ipntr[7];
    int ldh       = ncv;
    int ldq       = ncv;
    int iheigr    = bounds + ldh;
    int iheigi    = iheigr + ldh;
    int ihbds     = iheigi + ldh;
    int iuptri    = ihbds + ldh;
    int invsub    = iuptri + ldh * ncv;
    ipntr[8]      = iheigr;
    ipntr[9]      = iheigi;
    ipntr[10]     = ihbds;
    ipntr[11]     = iuptri;
    ipntr[12]     = invsub;

    int  irr      = ipntr[13] + ncv * ncv;
    int  iri      = irr + ncv;
    int  ibd      = iri + ncv;

    Real rnorm    = workl[ih + 2];
    workl[ih + 2] = Real(0);

    if (msglvl > 2) {
      detail::debug.vout(ncv, &workl[irr], "_neupd: Real part of Ritz values passed in from _NAUPD.");
      detail::debug.vout(ncv, &workl[iri], "_neupd: Imag part of Ritz values passed in from _NAUPD.");
      detail::debug.vout(ncv, &workl[ibd], "_neupd: Ritz estimates passed in from _NAUPD.");
    }

    std::vector<int> select(ncv, 0);

    if (rvec) {
      bool reord = false;

      for (int j = 0; j < ncv; j++) {
        workl[bounds + j] = Real(j + 1);
        select[j]         = 0;
      }

      {
        int np     = ncv - nev;
        int ishift = 0;
        int nev_t = nev, np_t = np;
        detail::ngets<Real>(ishift, which, nev_t, np_t, &workl[irr], &workl[iri], &workl[bounds], workl, &workl[np]);
      }

      if (msglvl > 2) {
        detail::debug.vout(ncv, &workl[irr], "_neupd: Real part of Ritz values after calling _NGETS.");
        detail::debug.vout(ncv, &workl[iri], "_neupd: Imag part of Ritz values after calling _NGETS.");
        detail::debug.vout(ncv, &workl[bounds], "_neupd: Ritz value indices after calling _NGETS.");
      }

      int numcnv = 0;
      for (int j = 1; j <= ncv; j++) {
        Real temp1 = std::max(eps23, detail::Ops<Real>::lapy2(workl[irr + ncv - j], workl[iri + ncv - j]));
        int  jj    = (int)workl[bounds + ncv - j];
        if (numcnv < nconv && workl[ibd + jj - 1] <= tol * temp1) {
          select[jj - 1] = 1;
          numcnv++;
          if (jj > nconv) reord = true;
        }
      }

      if (msglvl > 2) {
        detail::debug.ivout(1, &numcnv, "_neupd: Number of specified eigenvalues");
        detail::debug.ivout(1, &nconv, "_neupd: Number of \"converged\" eigenvalues");
      }

      if (numcnv != nconv) {
        info = -15;
        return;
      }

      detail::Ops<Real>::copy(ldh * ncv, &workl[ih], 1, &workl[iuptri], 1);
      {
        Real fzero = Real(0), fone = Real(1);
        detail::Ops<Real>::laset_("All", &ncv, &ncv, &fzero, &fone, &workl[invsub], &ldq);
      }

      {
        int ilo = 1, ihi = ncv;
        int wantt = 1, wantz = 1;
        detail::Ops<Real>::lahqr_(&wantt, &wantz, &ncv, &ilo, &ihi, &workl[iuptri], &ldh, &workl[iheigr], &workl[iheigi], &ilo,
                                  &ncv, &workl[invsub], &ldq, &ierr);
      }
      detail::Ops<Real>::copy(ncv, &workl[invsub + ncv - 1], ldq, &workl[ihbds], 1);

      if (ierr != 0) {
        info = -8;
        return;
      }

      if (msglvl > 1) {
        detail::debug.vout(ncv, &workl[iheigr], "_neupd: Real part of the eigenvalues of H");
        detail::debug.vout(ncv, &workl[iheigi], "_neupd: Imaginary part of the Eigenvalues of H");
        detail::debug.vout(ncv, &workl[ihbds], "_neupd: Last row of the Schur vector matrix");
      }

      if (reord) {
        int  nconv2;
        Real conds, sep;
        int  iwork_val;
        int  lwork_val = ncv, liwork_val = 1;
        detail::Ops<Real>::trsen_("None", "V", select.data(), &ncv, &workl[iuptri], &ldh, &workl[invsub], &ldq, &workl[iheigr],
                                  &workl[iheigi], &nconv2, &conds, &sep, &workl[ihbds], &lwork_val, &iwork_val, &liwork_val,
                                  &ierr);

        if (nconv2 < nconv) nconv = nconv2;

        if (ierr == 1) {
          info = 1;
          return;
        }

        if (msglvl > 2) {
          detail::debug.vout(ncv, &workl[iheigr], "_neupd: Real part of the eigenvalues of H--reordered");
          detail::debug.vout(ncv, &workl[iheigi], "_neupd: Imag part of the eigenvalues of H--reordered");
          if (msglvl > 3) {
            detail::debug.mout(ncv, ncv, &workl[iuptri], ldq, "_neupd: Quasi-triangular matrix after re-ordering");
          }
        }
      }

      detail::Ops<Real>::copy(ncv, &workl[invsub + ncv - 1], ldq, &workl[ihbds], 1);

      if (std::strcmp(type, "REGULR") == 0) {
        detail::Ops<Real>::copy(nconv, &workl[iheigr], 1, dr, 1);
        detail::Ops<Real>::copy(nconv, &workl[iheigi], 1, di, 1);
      }

      detail::Ops<Real>::geqr2_(&ncv, &nconv, &workl[invsub], &ldq, workev, &workev[ncv], &ierr);

      detail::Ops<Real>::orm2r_("Right", "Notranspose", &n, &ncv, &nconv, &workl[invsub], &ldq, workev, v, &ldv, &workd[n],
                                &ierr);
      detail::Ops<Real>::lacpy("All", n, nconv, v, ldv, z, ldz);

      for (int j = 0; j < nconv; j++) {
        if (workl[invsub + j * ldq + j] < Real(0)) {
          detail::Ops<Real>::scal(nconv, Real(-1), &workl[iuptri + j], ldq);
          detail::Ops<Real>::scal(nconv, Real(-1), &workl[iuptri + j * ldq], 1);
        }
      }

      if (howmny[0] == 'A') {
        for (int j = 0; j < ncv; j++) select[j] = (j < nconv) ? 1 : 0;

        {
          Real vl;
          int  outncv;
          detail::Ops<Real>::trevc_("Right", "Select", select.data(), &ncv, &workl[iuptri], &ldq, &vl, &ncv, &workl[invsub], &ldq,
                                    &ncv, &outncv, workev, &ierr);
        }

        if (ierr != 0) {
          info = -9;
          return;
        }

        int iconj = 0;
        for (int j = 0; j < nconv; j++) {
          if (workl[iheigi + j] == Real(0)) {
            Real temp = detail::Ops<Real>::nrm2(ncv, &workl[invsub + j * ldq], 1);
            detail::Ops<Real>::scal(ncv, Real(1) / temp, &workl[invsub + j * ldq], 1);
          } else if (iconj == 0) {
            Real temp = detail::Ops<Real>::lapy2(detail::Ops<Real>::nrm2(ncv, &workl[invsub + j * ldq], 1),
                                                 detail::Ops<Real>::nrm2(ncv, &workl[invsub + (j + 1) * ldq], 1));
            detail::Ops<Real>::scal(ncv, Real(1) / temp, &workl[invsub + j * ldq], 1);
            detail::Ops<Real>::scal(ncv, Real(1) / temp, &workl[invsub + (j + 1) * ldq], 1);
            iconj = 1;
          } else {
            iconj = 0;
          }
        }

        detail::Ops<Real>::gemv("T", ncv, nconv, Real(1), &workl[invsub], ldq, &workl[ihbds], 1, Real(0), workev, 1);

        iconj = 0;
        for (int j = 0; j < nconv; j++) {
          if (workl[iheigi + j] != Real(0)) {
            if (iconj == 0) {
              workev[j]     = detail::Ops<Real>::lapy2(workev[j], workev[j + 1]);
              workev[j + 1] = workev[j];
              iconj         = 1;
            } else {
              iconj = 0;
            }
          }
        }

        if (msglvl > 2) {
          detail::Ops<Real>::copy(ncv, &workl[invsub + ncv - 1], ldq, &workl[ihbds], 1);
          detail::debug.vout(ncv, &workl[ihbds], "_neupd: Last row of the eigenvector matrix for T");
          if (msglvl > 3) {
            detail::debug.mout(ncv, ncv, &workl[invsub], ldq, "_neupd: The eigenvector matrix for T");
          }
        }

        detail::Ops<Real>::copy(nconv, workev, 1, &workl[ihbds], 1);

        detail::Ops<Real>::geqr2_(&ncv, &nconv, &workl[invsub], &ldq, workev, &workev[ncv], &ierr);

        detail::Ops<Real>::orm2r_("Right", "Notranspose", &n, &ncv, &nconv, &workl[invsub], &ldq, workev, z, &ldz, &workd[n],
                                  &ierr);

        {
          Real done = Real(1);
          detail::Ops<Real>::trmm_("Right", "Upper", "No transpose", "Non-unit", &n, &nconv, &done, &workl[invsub], &ldq, z,
                                   &ldz);
        }
      }

    } else {
      detail::Ops<Real>::copy(nconv, &workl[ritzr_], 1, dr, 1);
      detail::Ops<Real>::copy(nconv, &workl[ritzi_], 1, di, 1);
      detail::Ops<Real>::copy(nconv, &workl[ritzr_], 1, &workl[iheigr], 1);
      detail::Ops<Real>::copy(nconv, &workl[ritzi_], 1, &workl[iheigi], 1);
      detail::Ops<Real>::copy(nconv, &workl[bounds], 1, &workl[ihbds], 1);
    }

    if (std::strcmp(type, "REGULR") == 0) {
      if (rvec) detail::Ops<Real>::scal(ncv, rnorm, &workl[ihbds], 1);
    } else {
      if (std::strcmp(type, "SHIFTI") == 0) {
        if (rvec) detail::Ops<Real>::scal(ncv, rnorm, &workl[ihbds], 1);

        for (int k = 0; k < ncv; k++) {
          Real temp        = detail::Ops<Real>::lapy2(workl[iheigr + k], workl[iheigi + k]);
          workl[ihbds + k] = std::abs(workl[ihbds + k]) / temp / temp;
        }
      }

      if (std::strcmp(type, "SHIFTI") == 0) {
        for (int k = 0; k < ncv; k++) {
          Real temp         = detail::Ops<Real>::lapy2(workl[iheigr + k], workl[iheigi + k]);
          workl[iheigr + k] = workl[iheigr + k] / temp / temp + sigmar;
          workl[iheigi + k] = -workl[iheigi + k] / temp / temp + sigmai;
        }
        detail::Ops<Real>::copy(nconv, &workl[iheigr], 1, dr, 1);
        detail::Ops<Real>::copy(nconv, &workl[iheigi], 1, di, 1);
      } else if (std::strcmp(type, "REALPT") == 0 || std::strcmp(type, "IMAGPT") == 0) {
        detail::Ops<Real>::copy(nconv, &workl[iheigr], 1, dr, 1);
        detail::Ops<Real>::copy(nconv, &workl[iheigi], 1, di, 1);
      }
    }

    if (std::strcmp(type, "SHIFTI") == 0 && msglvl > 1) {
      detail::debug.vout(nconv, dr, "_neupd: Untransformed real part of the Ritz values.");
      detail::debug.vout(nconv, di, "_neupd: Untransformed imag part of the Ritz values.");
      detail::debug.vout(nconv, &workl[ihbds], "_neupd: Ritz estimates of untransformed Ritz values.");
    } else if (std::strcmp(type, "REGULR") == 0 && msglvl > 1) {
      detail::debug.vout(nconv, dr, "_neupd: Real parts of converged Ritz values.");
      detail::debug.vout(nconv, di, "_neupd: Imag parts of converged Ritz values.");
      detail::debug.vout(nconv, &workl[ihbds], "_neupd: Associated Ritz estimates.");
    }

    if (std::strcmp(type, "SHIFTI") == 0 && rvec && howmny[0] == 'A') {
      int iconj = 0;
      for (int j = 0; j < nconv; j++) {
        if (workl[iheigi + j] == Real(0) && workl[iheigr + j] != Real(0)) {
          workev[j] = workl[invsub + j * ldq + ncv - 1] / workl[iheigr + j];
        } else if (iconj == 0) {
          Real temp = detail::Ops<Real>::lapy2(workl[iheigr + j], workl[iheigi + j]);
          if (temp != Real(0)) {
            workev[j] = (workl[invsub + j * ldq + ncv - 1] * workl[iheigr + j] +
                         workl[invsub + (j + 1) * ldq + ncv - 1] * workl[iheigi + j]) /
                        temp / temp;
            workev[j + 1] = (workl[invsub + (j + 1) * ldq + ncv - 1] * workl[iheigr + j] -
                             workl[invsub + j * ldq + ncv - 1] * workl[iheigi + j]) /
                            temp / temp;
          }
          iconj = 1;
        } else {
          iconj = 0;
        }
      }
      detail::Ops<Real>::ger(n, nconv, Real(1), resid, 1, workev, 1, z, ldz);
    }
  }

}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_NONSYM_HPP
