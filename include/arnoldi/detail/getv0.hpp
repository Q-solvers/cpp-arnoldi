#ifndef ARNOLDI_DETAIL_GETV0_HPP
#define ARNOLDI_DETAIL_GETV0_HPP

#include <arnoldi/detail/debug.hpp>
#include <arnoldi/detail/ops.hpp>
#include <arnoldi/detail/stats.hpp>

namespace arnoldi::detail {

  // getv0 — generate a starting vector (shared by sym / nonsym).
  // Replaces the reverse-communication version: callbacks op/bop are
  // invoked directly instead of returning ido.
  template <typename Scalar, typename OP, typename BOP, typename Comm>
  void getv0(const char* bmat, int itry, bool initv, int n, int j, Scalar* v, int ldv, Scalar* resid,
             detail::real_t<Scalar>& rnorm, Scalar* workd, int& ierr, OP&& op, BOP&& bop, const Comm& comm) {
    using Real          = detail::real_t<Scalar>;
    static int iseed[4] = {1, 3, 5, 7};
    int        msglvl   = detail::debug.getv0;
    double     t0, t1, t2, t3;

    detail::arscnd(t0);
    ierr = 0;

    if (!initv) {
      detail::Ops<Scalar>::larnv(2, iseed, n, resid);
    }

    if (itry == 1) {
      detail::stats.nopx++;
      detail::arscnd(t2);
      detail::Ops<Scalar>::copy(n, resid, 1, workd, 1);
      op(workd, &workd[n]);
      if (*bmat == 'G') {
        detail::arscnd(t3);
        detail::stats.mvopx += (t3 - t2);
      }
      detail::Ops<Scalar>::copy(n, &workd[n], 1, resid, 1);
    } else if (itry > 1 && *bmat == 'G') {
      detail::Ops<Scalar>::copy(n, resid, 1, &workd[n], 1);
    }

    detail::arscnd(t2);
    Real rnorm0 = detail::bnorm<Scalar>(*bmat, n, resid, workd, &workd[n], bop, comm);
    if (*bmat == 'G') {
      detail::arscnd(t3);
      detail::stats.mvbx += (t3 - t2);
    }
    rnorm = rnorm0;

    if (j != 1) {
      for (int iter = 0;; ++iter) {
        detail::Ops<Scalar>::gemv(detail::Ops<Scalar>::herm_trans(), n, j - 1, Scalar(1), v, ldv, workd, 1, Scalar(0), &workd[n],
                                  1);
        comm.allreduce_sum(&workd[n], j - 1);
        detail::Ops<Scalar>::gemv("N", n, j - 1, Scalar(-1), v, ldv, &workd[n], 1, Scalar(1), resid, 1);

        detail::arscnd(t2);
        rnorm = detail::bnorm<Scalar>(*bmat, n, resid, workd, &workd[n], bop, comm);
        if (*bmat == 'G') {
          detail::arscnd(t3);
          detail::stats.mvbx += (t3 - t2);
        }

        if (rnorm > Real(0.717) * rnorm0) break;

        rnorm0 = rnorm;
        if (iter >= 5) {
          for (int jj = 0; jj < n; jj++) resid[jj] = Scalar(0);
          rnorm = Real(0);
          ierr  = -1;
          break;
        }
      }
    }

    if (msglvl > 0) {
      detail::debug.vout(1, &rnorm, "_getv0: B-norm of initial / restarted starting vector");
    }
    if (msglvl > 3) {
      detail::debug.vout(n, resid, "_getv0: initial / restarted starting vector");
    }

    detail::arscnd(t1);
    detail::stats.getv0 += (t1 - t0);
  }

}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_GETV0_HPP
