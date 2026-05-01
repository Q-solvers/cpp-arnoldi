#ifndef ARNOLDI_DETAIL_SYM_HPP
#define ARNOLDI_DETAIL_SYM_HPP

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

  // saitr — Lanczos iteration (symmetric / Hermitian).
  // Scalar = double|float for real symmetric, complex<T> for Hermitian.
  // Vectors (v, resid, workd) are Scalar; tridiagonal h is detail::real_t<Scalar>.
  template <typename Scalar, typename OP, typename BOP, typename Comm>
  void saitr(const char* bmat, int n, int k, int np, int mode, Scalar* resid, detail::real_t<Scalar>& rnorm, Scalar* v, int ldv,
             detail::real_t<Scalar>* h, int ldh, Scalar* workd, int& info, OP&& op, BOP&& bop, const Comm& comm) {
    using Real    = detail::real_t<Scalar>;
    const int ipj = 0, irj = n, ivj = 2 * n;

    Real      safmin = detail::Ops<Scalar>::lamch("safmin");
    int       msglvl = detail::debug.aitr;
    double    t0, t1, t2, t3, t4, t5;
    detail::arscnd(t0);
    info = 0;

    for (int j = k + 1; j <= k + np; ++j) {
      if (msglvl > 2) {
        int jj = j;
        detail::debug.ivout(1, &jj, "_saitr: generating Arnoldi vector no.");
        detail::debug.vout(1, &rnorm, "_saitr: B-norm of the current residual =");
      }

      Real betaj;
      bool rstart = false;

      if (rnorm > Real(0)) {
        betaj = rnorm;
      } else {
        if (msglvl > 0) {
          int jj = j;
          detail::debug.ivout(1, &jj, "_saitr: ****** restart at step ******");
        }
        detail::stats.nrstrt++;
        rstart   = true;
        betaj    = Real(0);

        int ierr = -1;
        for (int itry = 1; itry <= 3; ++itry) {
          getv0<Scalar>(bmat, itry, false, n, j, v, ldv, resid, rnorm, workd, ierr, op, bop, comm);
          if (ierr >= 0) break;
        }
        if (ierr < 0) {
          info = j - 1;
          detail::arscnd(t1);
          detail::stats.aitr += (t1 - t0);
          return;
        }
      }

      detail::Ops<Scalar>::copy(n, resid, 1, &v[(j - 1) * ldv], 1);
      if (rnorm >= safmin) {
        Real temp1 = Real(1) / rnorm;
        detail::Ops<Scalar>::rscal(n, temp1, &v[(j - 1) * ldv], 1);
        detail::Ops<Scalar>::rscal(n, temp1, &workd[ipj], 1);
      } else {
        int  i_zero = 0, i_one = 1;
        int  infol;
        Real r_one = Real(1);
        detail::Ops<Scalar>::lascl_("G", &i_zero, &i_zero, &rnorm, &r_one, &n, &i_one, &v[(j - 1) * ldv], &n, &infol);
        detail::Ops<Scalar>::lascl_("G", &i_zero, &i_zero, &rnorm, &r_one, &n, &i_one, &workd[ipj], &n, &infol);
      }

      detail::stats.nopx++;
      detail::arscnd(t2);
      detail::Ops<Scalar>::copy(n, &v[(j - 1) * ldv], 1, &workd[ivj], 1);
      op(&workd[ivj], &workd[irj]);
      detail::arscnd(t3);
      detail::stats.mvopx += (t3 - t2);

      detail::Ops<Scalar>::copy(n, &workd[irj], 1, resid, 1);

      if (mode != 2) {
        detail::arscnd(t2);
        if (*bmat == 'G') {
          detail::stats.nbx++;
          bop(&workd[irj], &workd[ipj]);
        } else {
          detail::Ops<Scalar>::copy(n, resid, 1, &workd[ipj], 1);
        }
        if (*bmat == 'G') {
          detail::arscnd(t3);
          detail::stats.mvbx += (t3 - t2);
        }
      }

      Real wnorm;
      if (mode == 2) {
        wnorm = detail::prdotc<Scalar>(comm, n, resid, 1, &workd[ivj], 1);
        wnorm = std::sqrt(std::abs(wnorm));
      } else if (*bmat == 'G') {
        wnorm = detail::prdotc<Scalar>(comm, n, resid, 1, &workd[ipj], 1);
        wnorm = std::sqrt(std::abs(wnorm));
      } else {
        wnorm = detail::pnrm2<Scalar>(comm, n, resid, 1);
      }

      if (mode != 2) {
        detail::Ops<Scalar>::gemv(detail::Ops<Scalar>::herm_trans(), n, j, Scalar(1), v, ldv, &workd[ipj], 1, Scalar(0),
                                  &workd[irj], 1);
      } else {
        detail::Ops<Scalar>::gemv(detail::Ops<Scalar>::herm_trans(), n, j, Scalar(1), v, ldv, &workd[ivj], 1, Scalar(0),
                                  &workd[irj], 1);
      }
      comm.allreduce_sum(&workd[irj], j);
      detail::Ops<Scalar>::gemv("N", n, j, Scalar(-1), v, ldv, &workd[irj], 1, Scalar(1), resid, 1);

      h[ldh + j - 1] = std::real(workd[irj + j - 1]);
      if (j == 1 || rstart)
        h[j - 1] = Real(0);
      else
        h[j - 1] = rnorm;

      detail::arscnd(t4);

      detail::arscnd(t2);
      if (*bmat == 'G') {
        detail::stats.nbx++;
        detail::Ops<Scalar>::copy(n, resid, 1, &workd[irj], 1);
        bop(&workd[irj], &workd[ipj]);
      } else {
        detail::Ops<Scalar>::copy(n, resid, 1, &workd[ipj], 1);
      }
      if (*bmat == 'G') {
        detail::arscnd(t3);
        detail::stats.mvbx += (t3 - t2);
      }

      if (*bmat == 'G') {
        rnorm = detail::prdotc<Scalar>(comm, n, resid, 1, &workd[ipj], 1);
        rnorm = std::sqrt(std::abs(rnorm));
      } else {
        rnorm = detail::pnrm2<Scalar>(comm, n, resid, 1);
      }

      if (rnorm <= Real(0.717) * wnorm) {
        detail::stats.nrorth++;
        for (int riter = 0; riter <= 1; ++riter) {
          if (msglvl > 2) {
            Real xtemp[2] = {wnorm, rnorm};
            detail::debug.vout(2, xtemp, "_saitr: re-orthonalization ; wnorm and rnorm are");
          }

          detail::Ops<Scalar>::gemv(detail::Ops<Scalar>::herm_trans(), n, j, Scalar(1), v, ldv, &workd[ipj], 1, Scalar(0),
                                    &workd[irj], 1);
          comm.allreduce_sum(&workd[irj], j);
          detail::Ops<Scalar>::gemv("N", n, j, Scalar(-1), v, ldv, &workd[irj], 1, Scalar(1), resid, 1);

          if (j == 1 || rstart) h[j - 1] = Real(0);
          h[ldh + j - 1] = h[ldh + j - 1] + std::real(workd[irj + j - 1]);

          detail::arscnd(t2);
          Real rnorm1;
          if (*bmat == 'G') {
            detail::stats.nbx++;
            detail::Ops<Scalar>::copy(n, resid, 1, &workd[irj], 1);
            bop(&workd[irj], &workd[ipj]);
          } else {
            detail::Ops<Scalar>::copy(n, resid, 1, &workd[ipj], 1);
          }
          if (*bmat == 'G') {
            detail::arscnd(t3);
            detail::stats.mvbx += (t3 - t2);
          }

          if (*bmat == 'G') {
            rnorm1 = detail::prdotc<Scalar>(comm, n, resid, 1, &workd[ipj], 1);
            rnorm1 = std::sqrt(std::abs(rnorm1));
          } else {
            rnorm1 = detail::pnrm2<Scalar>(comm, n, resid, 1);
          }

          if (msglvl > 0 && riter > 0) {
            int jj = j;
            detail::debug.ivout(1, &jj, "_saitr: Iterative refinement for Arnoldi residual");
            if (msglvl > 2) {
              Real xtemp2[2] = {rnorm, rnorm1};
              detail::debug.vout(2, xtemp2, "_saitr: iterative refinement ; rnorm and rnorm1 are");
            }
          }

          if (rnorm1 > Real(0.717) * rnorm) {
            rnorm = rnorm1;
            break;
          }
          detail::stats.nitref++;
          rnorm = rnorm1;
          if (riter == 1) {
            for (int jj = 0; jj < n; jj++) resid[jj] = Scalar(0);
            rnorm = Real(0);
          }
        }
      }

      rstart = false;
      detail::arscnd(t5);
      detail::stats.itref += (t5 - t4);

      if (h[j - 1] < Real(0)) {
        h[j - 1] = -h[j - 1];
        if (j < k + np)
          detail::Ops<Scalar>::rscal(n, Real(-1), &v[j * ldv], 1);
        else
          detail::Ops<Scalar>::rscal(n, Real(-1), resid, 1);
      }
    }

    detail::arscnd(t1);
    detail::stats.aitr += (t1 - t0);

    if (msglvl > 1) {
      int knp = k + np;
      detail::debug.vout(knp, &h[ldh], "_saitr: main diagonal of matrix H of step K+NP.");
      if (knp > 1) {
        detail::debug.vout(knp - 1, &h[1], "_saitr: sub diagonal of matrix H of step K+NP.");
      }
    }
  }

  // sapps — implicit QR shifts for symmetric/Hermitian Lanczos.
  // Scalar-aware: Givens rotations on Real h/q, final V update on Scalar.
  template <typename Scalar>
  void sapps(int n, int kev, int np, detail::real_t<Scalar>* shift, Scalar* v, int ldv, detail::real_t<Scalar>* h, int ldh,
             Scalar* resid, detail::real_t<Scalar>* q, int ldq, Scalar* workd) {
    using Real     = detail::real_t<Scalar>;
    const Real one = Real(1), zero = Real(0);

    Real       epsmch = detail::Ops<Real>::lamch("Epsilon-Machine");

    int        i, iend, istart, itop, j, jj, kplusp, msglvl;
    Real       a1, a2, a3, a4, big, c, f, g, r, s;
    double     t0, t1;
    itop = 1;

    detail::arscnd(t0);
    msglvl = detail::debug.apps;

    kplusp = kev + np;

    detail::Ops<Real>::laset_("A", &kplusp, &kplusp, &zero, &one, q, &ldq);

    if (np == 0) return;

    for (jj = 1; jj <= np; jj++) {
      istart = itop;

      for (;;) {
        iend = kplusp;
        for (i = istart; i <= kplusp - 1; i++) {
          big = std::abs(h[ldh + i - 1]) + std::abs(h[ldh + i]);
          if (h[i] <= epsmch * big) {
            if (msglvl > 0) {
              detail::debug.ivout(1, &i, "_sapps: deflation at row/column no.");
              detail::debug.ivout(1, &jj, "_sapps: occurred before shift number.");
              detail::debug.vout(1, &h[i], "_sapps: the corresponding off diagonal element");
            }
            h[i] = zero;
            iend = i;
            break;
          }
        }

        if (istart < iend) {
          f = h[ldh + istart - 1] - shift[jj - 1];
          g = h[istart];
          detail::Ops<Real>::lartg_(&f, &g, &c, &s, &r);

          a1                  = c * h[ldh + istart - 1] + s * h[istart];
          a2                  = c * h[istart] + s * h[ldh + istart];
          a4                  = c * h[ldh + istart] - s * h[istart];
          a3                  = c * h[istart] - s * h[ldh + istart - 1];
          h[ldh + istart - 1] = c * a1 + s * a2;
          h[ldh + istart]     = c * a4 - s * a3;
          h[istart]           = c * a3 + s * a4;

          for (j = 1; j <= std::min(istart + jj, kplusp); j++) {
            a1                              = c * q[(istart - 1) * ldq + (j - 1)] + s * q[istart * ldq + (j - 1)];
            q[istart * ldq + (j - 1)]       = -s * q[(istart - 1) * ldq + (j - 1)] + c * q[istart * ldq + (j - 1)];
            q[(istart - 1) * ldq + (j - 1)] = a1;
          }

          for (i = istart + 1; i <= iend - 1; i++) {
            f    = h[i - 1];
            g    = s * h[i];

            h[i] = c * h[i];
            detail::Ops<Real>::lartg_(&f, &g, &c, &s, &r);

            if (r < zero) {
              r = -r;
              c = -c;
              s = -s;
            }

            h[i - 1]       = r;

            a1             = c * h[ldh + i - 1] + s * h[i];
            a2             = c * h[i] + s * h[ldh + i];
            a3             = c * h[i] - s * h[ldh + i - 1];
            a4             = c * h[ldh + i] - s * h[i];

            h[ldh + i - 1] = c * a1 + s * a2;
            h[ldh + i]     = c * a4 - s * a3;
            h[i]           = c * a3 + s * a4;

            for (j = 1; j <= std::min(i + jj, kplusp); j++) {
              a1                         = c * q[(i - 1) * ldq + (j - 1)] + s * q[i * ldq + (j - 1)];
              q[i * ldq + (j - 1)]       = -s * q[(i - 1) * ldq + (j - 1)] + c * q[i * ldq + (j - 1)];
              q[(i - 1) * ldq + (j - 1)] = a1;
            }
          }
        }

        istart = iend + 1;

        if (h[iend - 1] < zero) {
          h[iend - 1] = -h[iend - 1];
          detail::Ops<Real>::scal(kplusp, -one, &q[(iend - 1) * ldq], 1);
        }

        if (iend >= kplusp) break;
      }

      while (itop <= kplusp - 1 && !(h[itop] > zero)) itop++;
    }

    for (i = itop; i <= kplusp - 1; i++) {
      big = std::abs(h[ldh + i - 1]) + std::abs(h[ldh + i]);
      if (h[i] <= epsmch * big) {
        if (msglvl > 0) {
          detail::debug.ivout(1, &i, "_sapps: deflation at row/column no.");
          detail::debug.vout(1, &h[i], "_sapps: the corresponding off diagonal element");
        }
        h[i] = zero;
      }
    }

    // V_new = V * Q and resid update.
    if (h[kev] > zero) detail::Ops<Scalar>::gemv_rv("N", n, kplusp, one, v, ldv, &q[kev * ldq], 1, zero, &workd[n], 1);

    for (i = 1; i <= kev; i++) {
      detail::Ops<Scalar>::gemv_rv("N", n, kplusp - i + 1, one, v, ldv, &q[(kev - i) * ldq], 1, zero, workd, 1);
      detail::Ops<Scalar>::copy(n, workd, 1, &v[(kplusp - i) * ldv], 1);
    }

    for (i = 1; i <= kev; i++) {
      detail::Ops<Scalar>::copy(n, &v[(np + i - 1) * ldv], 1, &v[(i - 1) * ldv], 1);
    }

    if (h[kev] > zero) detail::Ops<Scalar>::copy(n, &workd[n], 1, &v[kev * ldv], 1);

    detail::Ops<Scalar>::rscal(n, q[(kev - 1) * ldq + kplusp - 1], resid, 1);
    if (h[kev] > zero) detail::Ops<Scalar>::raxpy(n, h[kev], &v[kev * ldv], 1, resid, 1);

    if (msglvl > 1) {
      detail::debug.vout(1, &q[(kev - 1) * ldq + kplusp - 1], "_sapps: sigmak of the updated residual vector");
      detail::debug.vout(1, &h[kev], "_sapps: betak of the updated residual vector");
      detail::debug.vout(kev, &h[ldh], "_sapps: updated main diagonal of H for next iteration");
      if (kev > 1) {
        detail::debug.vout(kev - 1, &h[1], "_sapps: updated sub diagonal of H for next iteration");
      }
    }

    detail::arscnd(t1);
    detail::stats.apps += (t1 - t0);
  }

  // saup2 — main Lanczos iteration driver (symmetric / Hermitian).
  template <typename Scalar, typename OP, typename BOP, typename Comm>
  void saup2(const char* bmat, int n, const char* which, int& nev, int& np, detail::real_t<Scalar> tol, Scalar* resid, int mode,
             int iupd, int ishift, int& mxiter, Scalar* v, int ldv, detail::real_t<Scalar>* h, int ldh,
             detail::real_t<Scalar>* ritz, detail::real_t<Scalar>* bounds, detail::real_t<Scalar>* q, int ldq,
             detail::real_t<Scalar>* workl, Scalar* workd, int& info, OP&& op, BOP&& bop, const Comm& comm) {
    using Real = detail::real_t<Scalar>;
    double t0, t1, t2, t3;
    detail::arscnd(t0);
    int  msglvl = detail::debug.aup2;

    Real eps23  = detail::Ops<Real>::lamch("Epsilon-Machine");
    eps23       = std::pow(eps23, Real(2.0 / 3.0));

    int  nev0   = nev;
    int  np0    = np;
    int  kplusp = nev0 + np0;
    int  nconv  = 0;
    int  iter   = 0;
    Real rnorm  = 0;
    int  ierr;
    char wprime[3];

    bool initv = (info != 0);
    if (initv) info = 0;

    const auto end_saup2 = [&]() {
      detail::arscnd(t1);
      detail::stats.aup2 = t1 - t0;
    };

    getv0<Scalar>(bmat, 1, initv, n, 1, v, ldv, resid, rnorm, workd, info, op, bop, comm);
    if (rnorm == Real(0)) {
      info = -9;
      end_saup2();
      return;
    }

    saitr<Scalar>(bmat, n, 0, nev0, mode, resid, rnorm, v, ldv, h, ldh, workd, info, op, bop, comm);
    if (info > 0) {
      np     = info;
      mxiter = iter;
      info   = -9999;
      end_saup2();
      return;
    }

    for (iter = 1; iter <= mxiter; ++iter) {
      if (msglvl > 0) {
        int it = iter;
        detail::debug.ivout(1, &it, "_saup2: **** Start of major iteration number ****");
      }
      if (msglvl > 1) {
        detail::debug.ivout(1, &nev, "_saup2: The length of the current Lanczos factorization");
        detail::debug.ivout(1, &np, "_saup2: Extend the Lanczos factorization by");
      }

      saitr<Scalar>(bmat, n, nev, np, mode, resid, rnorm, v, ldv, h, ldh, workd, info, op, bop, comm);
      if (info > 0) {
        np     = info;
        mxiter = iter;
        info   = -9999;
        end_saup2();
        return;
      }

      if (msglvl > 1) {
        detail::debug.vout(1, &rnorm, "_saup2: Current B-norm of residual for factorization");
      }

      detail::seigt<Real>(rnorm, kplusp, h, ldh, ritz, bounds, workl, ierr);
      if (ierr != 0) {
        info = -8;
        end_saup2();
        return;
      }

      detail::Ops<Real>::copy(kplusp, ritz, 1, &workl[kplusp], 1);
      detail::Ops<Real>::copy(kplusp, bounds, 1, &workl[2 * kplusp], 1);

      nev = nev0;
      np  = np0;
      detail::sgets<Real>(ishift, which, nev, np, ritz, bounds, workl);

      detail::Ops<Real>::copy(nev, &bounds[np], 1, &workl[np], 1);
      detail::sconv<Real>(nev, &ritz[np], &workl[np], tol, nconv);

      if (msglvl > 2) {
        int kp[3] = {nev, np, nconv};
        detail::debug.ivout(3, kp, "_saup2: NEV, NP, NCONV are");
        detail::debug.vout(kplusp, ritz, "_saup2: The eigenvalues of H");
        detail::debug.vout(kplusp, bounds, "_saup2: Ritz estimates of the current NCV Ritz values");
      }

      {
        int nptemp = np;
        for (int j = 0; j < nptemp; j++) {
          if (bounds[j] == Real(0)) {
            np--;
            nev++;
          }
        }
      }

      if ((nconv >= nev0) || (iter > mxiter) || (np == 0)) {
        if (std::strcmp(which, "BE") == 0) {
          std::strcpy(wprime, "SA");
          detail::sortr<Real>(wprime, true, kplusp, ritz, bounds);
          int nevd2 = nev0 / 2;
          int nevm2 = nev0 - nevd2;
          if (nev > 1) {
            int npp = kplusp - nev0;
            detail::Ops<Real>::swap(std::min(nevd2, npp), &ritz[nevm2], 1, &ritz[std::max(kplusp - nevd2, kplusp - npp)], 1);
            detail::Ops<Real>::swap(std::min(nevd2, npp), &bounds[nevm2], 1, &bounds[std::max(kplusp - nevd2, kplusp - npp)], 1);
          }
        } else {
          if (std::strcmp(which, "LM") == 0) std::strcpy(wprime, "SM");
          if (std::strcmp(which, "SM") == 0) std::strcpy(wprime, "LM");
          if (std::strcmp(which, "LA") == 0) std::strcpy(wprime, "SA");
          if (std::strcmp(which, "SA") == 0) std::strcpy(wprime, "LA");
          detail::sortr<Real>(wprime, true, kplusp, ritz, bounds);
        }

        for (int j = 0; j < nev0; j++) {
          Real temp = std::max(eps23, std::abs(ritz[j]));
          bounds[j] = bounds[j] / temp;
        }
        std::strcpy(wprime, "LA");
        detail::sortr<Real>(wprime, true, nev0, bounds, ritz);
        for (int j = 0; j < nev0; j++) {
          Real temp = std::max(eps23, std::abs(ritz[j]));
          bounds[j] = bounds[j] * temp;
        }

        if (std::strcmp(which, "BE") == 0) {
          std::strcpy(wprime, "LA");
          detail::sortr<Real>(wprime, true, nconv, ritz, bounds);
        } else {
          detail::sortr<Real>(which, true, nconv, ritz, bounds);
        }

        h[0] = rnorm;

        if (msglvl > 1) {
          detail::debug.vout(kplusp, ritz, "_saup2: Sorted Ritz values.");
          detail::debug.vout(kplusp, bounds, "_saup2: Sorted ritz estimates.");
        }

        if (iter > mxiter && nconv < nev) info = 1;
        if (np == 0 && nconv < nev0) info = 2;

        np     = nconv;
        mxiter = iter;
        nev    = nconv;
        end_saup2();
        return;

      } else if (nconv < nev && ishift == 1) {
        int nevbef = nev;
        nev        = nev + std::min(nconv, np / 2);
        if (nev == 1 && kplusp >= 6)
          nev = kplusp / 2;
        else if (nev == 1 && kplusp > 2)
          nev = 2;
        np = kplusp - nev;

        if (nevbef < nev) detail::sgets<Real>(ishift, which, nev, np, ritz, bounds, workl);
      }

      if (msglvl > 0) {
        detail::debug.ivout(1, &nconv, "_saup2: no. of \"converged\" Ritz values at this iter.");
        if (msglvl > 1) {
          int kp[2] = {nev, np};
          detail::debug.ivout(2, kp, "_saup2: NEV and NP are");
          detail::debug.vout(nev, &ritz[np], "_saup2: \"wanted\" Ritz values.");
          detail::debug.vout(nev, &bounds[np], "_saup2: Ritz estimates of the \"wanted\" values ");
        }
      }

      sapps<Scalar>(n, nev, np, ritz, v, ldv, h, ldh, resid, q, ldq, workd);

      detail::arscnd(t2);
      if (*bmat == 'G') {
        detail::stats.nbx++;
        detail::Ops<Scalar>::copy(n, resid, 1, &workd[n], 1);
        bop(&workd[n], workd);
        rnorm = detail::prdotc<Scalar>(comm, n, resid, 1, workd, 1);
        rnorm = std::sqrt(std::abs(rnorm));
      } else {
        detail::Ops<Scalar>::copy(n, resid, 1, workd, 1);
        rnorm = detail::pnrm2<Scalar>(comm, n, resid, 1);
      }
      if (*bmat == 'G') {
        detail::arscnd(t3);
        detail::stats.mvbx += (t3 - t2);
      }

      if (msglvl > 2) {
        detail::debug.vout(1, &rnorm, "_saup2: B-norm of residual for NEV factorization");
        detail::debug.vout(nev, &h[ldh], "_saup2: main diagonal of compressed H matrix");
        if (nev > 1) {
          detail::debug.vout(nev - 1, &h[1], "_saup2: subdiagonal of compressed H matrix");
        }
      }
    }

    // Exceeded mxiter.
    info = 1;
    end_saup2();
  }

  // saupd — symmetric/Hermitian eigensolver entry point (callback).
  // Scalar = double|float for real, complex<T> for Hermitian.
  // workd is Scalar* (3n), workl is detail::real_t<Scalar>* (ncv^2 + 8*ncv).
  template <typename Scalar, typename OP, typename BOP, typename Comm>
  void saupd(const char* bmat, int n, const char* which, int nev, detail::real_t<Scalar>& tol, Scalar* resid, int ncv, Scalar* v,
             int ldv, int* iparam, int* ipntr, Scalar* workd, detail::real_t<Scalar>* workl, int lworkl, int& info, OP&& op,
             BOP&& bop, const Comm& comm) {
    using Real = detail::real_t<Scalar>;
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
    else if (ncv <= nev)
      ierr = -3;
    else if (mxiter <= 0)
      ierr = -4;
    else if (std::strcmp(which, "LM") != 0 && std::strcmp(which, "SM") != 0 && std::strcmp(which, "LA") != 0 &&
             std::strcmp(which, "SA") != 0 && std::strcmp(which, "BE") != 0)
      ierr = -5;
    else if (bmat[0] != 'I' && bmat[0] != 'G')
      ierr = -6;
    else if (lworkl < ncv * ncv + 8 * ncv)
      ierr = -7;
    else if (mode < 1 || mode > 5)
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
    if (tol <= Real(0)) tol = detail::Ops<Real>::lamch("EpsMach");

    int np   = ncv - nev;
    int nev0 = nev;

    for (int j = 0; j < ncv * ncv + 8 * ncv; j++) workl[j] = Real(0);

    int ldh    = ncv;
    int ldq    = ncv;
    int ih     = 0;
    int ritz   = ih + 2 * ldh;
    int bounds = ritz + ncv;
    int iq     = bounds + ncv;
    int iw     = iq + ncv * ncv;
    int next   = iw + 3 * ncv;

    ipntr[3]   = next + 1;
    ipntr[4]   = ih + 1;
    ipntr[5]   = ritz + 1;
    ipntr[6]   = bounds + 1;
    ipntr[10]  = iw + 1;

    saup2<Scalar>(bmat, n, which, nev0, np, tol, resid, mode, iupd, ishift, mxiter, v, ldv, &workl[ih], ldh, &workl[ritz],
                  &workl[bounds], &workl[iq], ldq, &workl[iw], workd, info, op, bop, comm);

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
          "     = Symmetric implicit Arnoldi update code    =\n"
          "     = Version Number:  2.4  (callback)         =\n"
          "     =============================================\n"
          "     Total number update iterations             = %5d\n"
          "     Total number of OP*x operations            = %5d\n"
          "     Total number of B*x operations             = %5d\n"
          "     Total number of reorthogonalization steps  = %5d\n"
          "     Total number of iterative refinement steps = %5d\n"
          "     Total number of restart steps              = %5d\n",
          mxiter, detail::stats.nopx, detail::stats.nbx, detail::stats.nrorth, detail::stats.nitref, detail::stats.nrstrt);
    }

    detail::arscnd(t1);
    detail::stats.aupd = t1 - t0;
  }

  // op + bop, no comm (defaults to SerialComm).
  template <typename Scalar, typename OP, typename BOP>
  void saupd(const char* bmat, int n, const char* which, int nev, detail::real_t<Scalar>& tol, Scalar* resid, int ncv, Scalar* v,
             int ldv, int* iparam, int* ipntr, Scalar* workd, detail::real_t<Scalar>* workl, int lworkl, int& info, OP&& op,
             BOP&& bop) {
    saupd<Scalar>(bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, std::forward<OP>(op),
                  std::forward<BOP>(bop), SerialComm{});
  }

  // Standard problem (bmat='I'), no bop, no comm.
  template <typename Scalar, typename OP>
  void saupd(const char* bmat, int n, const char* which, int nev, detail::real_t<Scalar>& tol, Scalar* resid, int ncv, Scalar* v,
             int ldv, int* iparam, int* ipntr, Scalar* workd, detail::real_t<Scalar>* workl, int lworkl, int& info, OP&& op) {
    saupd<Scalar>(
        bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info, std::forward<OP>(op),
        [](const Scalar*, Scalar*) {}, SerialComm{});
  }

  // seupd — symmetric/Hermitian eigenvector extraction (callback).
  // d is Real* (eigenvalues), z is Scalar* (eigenvectors).
  template <typename Scalar, typename Comm>
  void seupd(bool rvec, const char* howmny, detail::real_t<Scalar>* d, Scalar* z, int ldz, detail::real_t<Scalar> sigma,
             const char* bmat, int n, const char* which, int nev, detail::real_t<Scalar> tol, Scalar* resid, int ncv, Scalar* v,
             int ldv, int* iparam, int* ipntr, Scalar* workd, detail::real_t<Scalar>* workl, int lworkl, int& info,
             const Comm& comm) {
    using Real = detail::real_t<Scalar>;

    char type[7];
    int  bounds, ierr, ih, ihb, ihd, iq, iw, j, k, ldh, ldq, mode, msglvl, nconv, next, ritz, irz, ibd, np, ishift, leftptr,
        rghtptr, numcnv, jj;
    Real bnorm2, rnorm, temp, temp1, eps23;
    bool reord;

    msglvl = detail::debug.eupd;
    mode   = iparam[6];
    nconv  = iparam[4];
    info   = 0;

    if (nconv == 0) return;
    ierr = 0;

    if (nconv <= 0) ierr = -14;
    if (n <= 0) ierr = -1;
    if (nev <= 0) ierr = -2;
    if (ncv <= nev) ierr = -3;
    if (std::strcmp(which, "LM") != 0 && std::strcmp(which, "SM") != 0 && std::strcmp(which, "LA") != 0 &&
        std::strcmp(which, "SA") != 0 && std::strcmp(which, "BE") != 0)
      ierr = -5;
    if (*bmat != 'I' && *bmat != 'G') ierr = -6;
    if (rvec && lworkl < ncv * ncv + 8 * ncv) ierr = -7;
    if (rvec && *howmny != 'A' && *howmny != 'P' && *howmny != 'S') ierr = -15;
    if (rvec && *howmny == 'S') ierr = -16;

    if (mode == 1 || mode == 2)
      std::strcpy(type, "REGULR");
    else if (mode == 3)
      std::strcpy(type, "SHIFTI");
    else if (mode == 4)
      std::strcpy(type, "BUCKLE");
    else if (mode == 5)
      std::strcpy(type, "CAYLEY");
    else
      ierr = -10;
    if (mode == 1 && *bmat == 'G') ierr = -11;

    if (ierr != 0) {
      info = ierr;
      return;
    }

    ih       = ipntr[4] - 1;
    ritz     = ipntr[5] - 1;
    bounds   = ipntr[6] - 1;
    ldh      = ncv;
    ldq      = ncv;
    ihd      = bounds + ldh;
    ihb      = ihd + ldh;
    iq       = ihb + ldh;
    iw       = iq + ldh * ncv;
    next     = iw + 2 * ncv;
    ipntr[3] = next + 1;
    ipntr[7] = ihd + 1;
    ipntr[8] = ihb + 1;
    ipntr[9] = iq + 1;

    irz      = (ipntr[10] - 1) + ncv;
    ibd      = irz + ncv;

    eps23    = detail::Ops<Real>::lamch("Epsilon-Machine");
    eps23    = std::pow(eps23, Real(2.0 / 3.0));

    rnorm    = workl[ih];
    if (*bmat == 'I')
      bnorm2 = rnorm;
    else
      bnorm2 = detail::pnrm2<Scalar>(comm, n, workd, 1);

    std::vector<int> select(ncv, 0);

    if (rvec) {
      reord = false;

      for (j = 1; j <= ncv; j++) {
        workl[bounds + j - 1] = (Real)j;
        select[j - 1]         = 0;
      }

      np     = ncv - nev;
      ishift = 0;
      detail::sgets<Real>(ishift, which, nev, np, &workl[irz], &workl[bounds], workl);

      numcnv = 0;
      for (j = 1; j <= ncv; j++) {
        temp1 = std::max(eps23, std::abs(workl[irz + ncv - j]));
        jj    = (int)workl[bounds + ncv - j];
        if (numcnv < nconv && workl[ibd + jj - 1] <= tol * temp1) {
          select[jj - 1] = 1;
          numcnv++;
          if (jj > nconv) reord = true;
        }
      }

      if (numcnv != nconv) {
        info = -17;
        return;
      }

      detail::Ops<Real>::copy(ncv - 1, &workl[ih + 1], 1, &workl[ihb], 1);
      detail::Ops<Real>::copy(ncv, &workl[ih + ldh], 1, &workl[ihd], 1);

      detail::Ops<Real>::steqr("I", ncv, &workl[ihd], &workl[ihb], &workl[iq], ldq, &workl[iw], ierr);

      if (ierr != 0) {
        info = -8;
        return;
      }

      if (reord) {
        leftptr = 1;
        rghtptr = ncv;
        while (leftptr < rghtptr) {
          if (select[leftptr - 1]) {
            leftptr++;
          } else if (!select[rghtptr - 1]) {
            rghtptr--;
          } else {
            temp                     = workl[ihd + leftptr - 1];
            workl[ihd + leftptr - 1] = workl[ihd + rghtptr - 1];
            workl[ihd + rghtptr - 1] = temp;
            detail::Ops<Real>::copy(ncv, &workl[iq + ncv * (leftptr - 1)], 1, &workl[iw], 1);
            detail::Ops<Real>::copy(ncv, &workl[iq + ncv * (rghtptr - 1)], 1, &workl[iq + ncv * (leftptr - 1)], 1);
            detail::Ops<Real>::copy(ncv, &workl[iw], 1, &workl[iq + ncv * (rghtptr - 1)], 1);
            leftptr++;
            rghtptr--;
          }
        }
      }

      detail::Ops<Real>::copy(nconv, &workl[ihd], 1, d, 1);

    } else {
      detail::Ops<Real>::copy(nconv, &workl[ritz], 1, d, 1);
      detail::Ops<Real>::copy(ncv, &workl[ritz], 1, &workl[ihd], 1);
    }

    if (std::strcmp(type, "REGULR") == 0) {
      if (rvec) {
        detail::sesrt<Real>("LA", rvec, nconv, d, ncv, &workl[iq], ldq);
      } else {
        detail::Ops<Real>::copy(ncv, &workl[bounds], 1, &workl[ihb], 1);
      }
    } else {
      detail::Ops<Real>::copy(ncv, &workl[ihd], 1, &workl[iw], 1);
      if (std::strcmp(type, "SHIFTI") == 0) {
        for (k = 1; k <= ncv; k++) workl[ihd + k - 1] = Real(1) / workl[ihd + k - 1] + sigma;
      } else if (std::strcmp(type, "BUCKLE") == 0) {
        for (k = 1; k <= ncv; k++) workl[ihd + k - 1] = sigma * workl[ihd + k - 1] / (workl[ihd + k - 1] - Real(1));
      } else if (std::strcmp(type, "CAYLEY") == 0) {
        for (k = 1; k <= ncv; k++) workl[ihd + k - 1] = sigma * (workl[ihd + k - 1] + Real(1)) / (workl[ihd + k - 1] - Real(1));
      }
      detail::Ops<Real>::copy(nconv, &workl[ihd], 1, d, 1);
      detail::sortr<Real>("LA", true, nconv, &workl[ihd], &workl[iw]);
      if (rvec) {
        detail::sesrt<Real>("LA", rvec, nconv, d, ncv, &workl[iq], ldq);
      } else {
        detail::Ops<Real>::copy(ncv, &workl[bounds], 1, &workl[ihb], 1);
        detail::Ops<Real>::scal(ncv, bnorm2 / rnorm, &workl[ihb], 1);
        detail::sortr<Real>("LA", true, nconv, d, &workl[ihb]);
      }
    }

    if (rvec && *howmny == 'A') {
      // z = V * S (Scalar V times Real eigenvector matrix S). When z
      // aliases v, a column-by-column multiply would destroy V columns
      // before they are fully consumed; use a temporary in that case.
      Scalar*             vbuf = v;
      std::vector<Scalar> tmpbuf;
      if (z == v) {
        tmpbuf.resize(static_cast<std::size_t>(ldv) * ncv);
        detail::Ops<Scalar>::copy(ldv * ncv, v, 1, tmpbuf.data(), 1);
        vbuf = tmpbuf.data();
      }
      for (j = 0; j < nconv; j++) {
        for (int ii = 0; ii < n; ii++) z[j * ldz + ii] = Scalar(0);
        for (k = 0; k < ncv; k++) {
          detail::Ops<Scalar>::raxpy(n, workl[iq + j * ldq + k], &vbuf[k * ldv], 1, &z[j * ldz], 1);
        }
      }

      // Last-row weights for error bounds.
      for (j = 1; j <= ncv - 1; j++) workl[ihb + j - 1] = Real(0);
      workl[ihb + ncv - 1] = Real(1);
      for (j = 0; j < nconv; j++) {
        Real wt = Real(0);
        for (k = 0; k < ncv; k++) wt += workl[iq + j * ldq + k] * workl[ihb + k];
        workl[iw + ncv + j] = wt;
      }
    }

    if (std::strcmp(type, "REGULR") == 0 && rvec) {
      for (j = 1; j <= ncv; j++) workl[ihb + j - 1] = rnorm * std::abs(workl[iw + ncv + j - 1]);
    } else if (std::strcmp(type, "REGULR") != 0 && rvec) {
      detail::Ops<Real>::scal(ncv, bnorm2, &workl[ihb], 1);
      if (std::strcmp(type, "SHIFTI") == 0) {
        for (k = 1; k <= ncv; k++) workl[ihb + k - 1] = std::abs(workl[ihb + k - 1]) / (workl[iw + k - 1] * workl[iw + k - 1]);
      } else if (std::strcmp(type, "BUCKLE") == 0) {
        for (k = 1; k <= ncv; k++)
          workl[ihb + k - 1] =
              sigma * std::abs(workl[ihb + k - 1]) / ((workl[iw + k - 1] - Real(1)) * (workl[iw + k - 1] - Real(1)));
      } else if (std::strcmp(type, "CAYLEY") == 0) {
        for (k = 1; k <= ncv; k++)
          workl[ihb + k - 1] = std::abs(workl[ihb + k - 1] / (workl[iw + k - 1] * (workl[iw + k - 1] - Real(1))));
      }
    }

    if (rvec && (std::strcmp(type, "SHIFTI") == 0 || std::strcmp(type, "CAYLEY") == 0)) {
      for (k = 0; k < nconv; k++) workl[iw + k] = workl[iw + ncv + k] / workl[iw + k];
    } else if (rvec && std::strcmp(type, "BUCKLE") == 0) {
      for (k = 0; k < nconv; k++) workl[iw + k] = workl[iw + ncv + k] / (workl[iw + k] - Real(1));
    }

    if (rvec && std::strcmp(type, "REGULR") != 0) {
      for (k = 0; k < nconv; k++) {
        detail::Ops<Scalar>::raxpy(n, workl[iw + k], resid, 1, &z[k * ldz], 1);
      }
    }
  }

  // seupd without Comm (defaults to SerialComm).
  template <typename Scalar>
  void seupd(bool rvec, const char* howmny, detail::real_t<Scalar>* d, Scalar* z, int ldz, detail::real_t<Scalar> sigma,
             const char* bmat, int n, const char* which, int nev, detail::real_t<Scalar> tol, Scalar* resid, int ncv, Scalar* v,
             int ldv, int* iparam, int* ipntr, Scalar* workd, detail::real_t<Scalar>* workl, int lworkl, int& info) {
    seupd<Scalar>(rvec, howmny, d, z, ldz, sigma, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl,
                  lworkl, info, SerialComm{});
  }

}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_SYM_HPP
