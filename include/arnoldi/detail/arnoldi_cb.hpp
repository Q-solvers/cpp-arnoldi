#ifndef ARNOLDI_DETAIL_ARNOLDI_CB_HPP
#define ARNOLDI_DETAIL_ARNOLDI_CB_HPP

// Callback-based ARPACK — no reverse communication; control flow uses structured
// loops and early returns.
//
// Usage (standard eigenvalue problem, bmat='I'):
//
//   auto op = [&](const double* x, double* y) { matvec(n, x, y); };
//   arnoldi::detail::naupd<double>("I", n, "SM", nev, tol, resid, ncv,
//                             v, n, iparam, ipntr, workd, workl, lworkl, info, op);
//   arnoldi::detail::neupd<double>(...);      // nonsymmetric extraction
//   arnoldi::detail::seupd<double>(...);      // symmetric / Hermitian extraction
//
// For generalized problems (bmat='G'), supply a second callback for B*x:
//
//   arnoldi::detail::naupd<double>("G", ..., info, op, bop);

#include <arnoldi/detail/getv0.hpp>
#include <arnoldi/detail/nonsym.hpp>
#include <arnoldi/detail/sym.hpp>

#endif  // ARNOLDI_DETAIL_ARNOLDI_CB_HPP
