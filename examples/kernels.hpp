#ifndef ARPACK_CALLBACK_EXAMPLES_KERNELS_HPP
#define ARPACK_CALLBACK_EXAMPLES_KERNELS_HPP

// Matrix-vector kernels matching Fortran EXAMPLES/ reference drivers.

#include <arnoldi/detail/ops.hpp>

namespace arnoldi_examples {

  template <typename Real>
  void tv_sym_row(int nx, const Real* x, Real* y) {
    Real four = 4, neg_one = -1;
    y[0] = four * x[0] + neg_one * x[1];
    for (int j = 1; j < nx - 1; j++) y[j] = neg_one * x[j - 1] + four * x[j] + neg_one * x[j + 1];
    y[nx - 1] = neg_one * x[nx - 2] + four * x[nx - 1];
  }

  template <typename Real>
  void av_sym_laplacian_2d(int nx, int n, const Real* v, Real* w) {
    using RO = arnoldi::detail::Ops<Real>;
    Real one = 1;
    tv_sym_row(nx, v, w);
    RO::axpy(nx, -one, v + nx, 1, w, 1);
    for (int j = 2; j < nx - 1; j++) {
      int lo = (j - 1) * nx;
      tv_sym_row(nx, v + lo, w + lo);
      RO::axpy(nx, -one, v + lo - nx, 1, w + lo, 1);
      RO::axpy(nx, -one, v + lo + nx, 1, w + lo, 1);
    }
    int lo = (nx - 1) * nx;
    tv_sym_row(nx, v + lo, w + lo);
    RO::axpy(nx, -one, v + lo - nx, 1, w + lo, 1);
    Real h2 = one / static_cast<Real>((nx + 1) * (nx + 1));
    RO::scal(n, one / h2, w, 1);
  }

  template <typename Real>
  void tv_cd_row(int nx, Real rho, const Real* x, Real* y) {
    using RO = arnoldi::detail::Ops<Real>;
    Real one = 1, half = static_cast<Real>(0.5);
    Real h  = one / static_cast<Real>(nx + 1);
    Real h2 = h * h;
    Real dd = static_cast<Real>(4) / h2;
    Real dl = -one / h2 - half * rho / h;
    Real du = -one / h2 + half * rho / h;
    y[0]    = dd * x[0] + du * x[1];
    for (int j = 1; j < nx - 1; j++) y[j] = dl * x[j - 1] + dd * x[j] + du * x[j + 1];
    y[nx - 1] = dl * x[nx - 2] + dd * x[nx - 1];
  }

  template <typename Real>
  void av_conv_diff_2d(int nx, int n, Real rho, const Real* v, Real* w) {
    using RO = arnoldi::detail::Ops<Real>;
    Real one = 1;
    Real h2  = one / static_cast<Real>((nx + 1) * (nx + 1));
    tv_cd_row(nx, rho, v, w);
    RO::axpy(nx, -one / h2, v + nx, 1, w, 1);
    for (int j = 2; j < nx - 1; j++) {
      int lo = (j - 1) * nx;
      tv_cd_row(nx, rho, v + lo, w + lo);
      RO::axpy(nx, -one / h2, v + lo - nx, 1, w + lo, 1);
      RO::axpy(nx, -one / h2, v + lo + nx, 1, w + lo, 1);
    }
    {
      int lo = (nx - 1) * nx;
      tv_cd_row(nx, rho, v + lo, w + lo);
      RO::axpy(nx, -one / h2, v + lo - nx, 1, w + lo, 1);
    }
  }

  /// Standard 1D Laplacian (dsdrv2): Dirichlet, mesh size 1/(n+1).
  template <typename Real>
  void av_laplacian_1d(int n, const Real* v, Real* w) {
    Real one = 1, two = 2;
    Real h2 = one / static_cast<Real>((n + 1) * (n + 1));
    w[0]    = two * v[0] - v[1];
    for (int j = 1; j < n - 1; j++) w[j] = -v[j - 1] + two * v[j] - v[j + 1];
    w[n - 1] = -v[n - 2] + two * v[n - 1];
    for (int j = 0; j < n; j++) w[j] /= h2;
  }

  /// Stiffness A and mass M from EXAMPLES/SYM/dsdrv3.f (piecewise linear FE on [0,1]).
  template <typename Real>
  void fe_sym_stiffness_1d(int n, const Real* v, Real* w) {
    Real one = 1, two = 2;
    w[0] = two * v[0] - v[1];
    for (int j = 1; j < n - 1; j++) w[j] = -v[j - 1] + two * v[j] - v[j + 1];
    w[n - 1] = -v[n - 2] + two * v[n - 1];
    Real h   = one / static_cast<Real>(n + 1);
    using RO = arnoldi::detail::Ops<Real>;
    RO::scal(n, one / h, w, 1);
  }

  template <typename Real>
  void fe_sym_mass_1d(int n, const Real* v, Real* w) {
    Real one = 1, four = 4, six = 6;
    w[0] = four * v[0] + v[1];
    for (int j = 1; j < n - 1; j++) w[j] = v[j - 1] + four * v[j] + v[j + 1];
    w[n - 1] = v[n - 2] + four * v[n - 1];
    Real h   = one / (static_cast<Real>(n + 1) * six);
    using RO = arnoldi::detail::Ops<Real>;
    RO::scal(n, h, w, 1);
  }

  /// Stiffness A and mass M from EXAMPLES/NONSYM/dndrv3.f (FE convection–diffusion + mass).
  template <typename Real>
  void fe_ns_conv_diff_stiffness_1d(int n, Real rho, const Real* v, Real* w) {
    Real one = 1, two = 2;
    Real h  = one / static_cast<Real>(n + 1);
    Real s  = rho / two;
    Real dd = two / h;
    Real dl = -one / h - s;
    Real du = -one / h + s;
    w[0]    = dd * v[0] + du * v[1];
    for (int j = 1; j < n - 1; j++) w[j] = dl * v[j - 1] + dd * v[j] + du * v[j + 1];
    w[n - 1] = dl * v[n - 2] + dd * v[n - 1];
  }

  template <typename Real>
  void fe_ns_mass_1d(int n, const Real* v, Real* w) {
    Real one = 1, four = 4;
    w[0] = four * v[0] + one * v[1];
    for (int j = 1; j < n - 1; j++) w[j] = one * v[j - 1] + four * v[j] + one * v[j + 1];
    w[n - 1] = one * v[n - 2] + four * v[n - 1];
    Real h   = one / static_cast<Real>(n + 1);
    using RO = arnoldi::detail::Ops<Real>;
    RO::scal(n, h, w, 1);
  }

}  // namespace arnoldi_examples

#endif
