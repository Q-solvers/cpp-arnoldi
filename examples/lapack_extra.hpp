#ifndef ARPACK_CALLBACK_EXAMPLES_LAPACK_EXTRA_HPP
#define ARPACK_CALLBACK_EXAMPLES_LAPACK_EXTRA_HPP

// Extra LAPACK declarations used by Fortran-derived examples (tridiagonal solvers).

#include <type_traits>

extern "C" {

void dgttrf_(const int* n, double* dl, double* d, double* du, double* du2, int* ipiv, int* info);
void dgttrs_(const char* trans, const int* n, const int* nrhs, const double* dl, const double* d, const double* du,
             const double* du2, const int* ipiv, double* b, const int* ldb, int* info);

void sgttrf_(const int* n, float* dl, float* d, float* du, float* du2, int* ipiv, int* info);
void sgttrs_(const char* trans, const int* n, const int* nrhs, const float* dl, const float* d, const float* du, const float* du2,
             const int* ipiv, float* b, const int* ldb, int* info);

void dpttrf_(const int* n, double* d, double* e, int* info);
void dpttrs_(const int* n, const int* nrhs, const double* d, const double* e, double* b, const int* ldb, int* info);

void spttrf_(const int* n, float* d, float* e, int* info);
void spttrs_(const int* n, const int* nrhs, const float* d, const float* e, float* b, const int* ldb, int* info);

}  // extern "C"

namespace arnoldi_examples::lapackx {

  template <typename Real>
  int gttrf(int n, Real* dl, Real* d, Real* du, Real* du2, int* ipiv) {
    int info = 0;
    if constexpr (std::is_same_v<Real, double>) {
      dgttrf_(&n, dl, d, du, du2, ipiv, &info);
    } else {
      sgttrf_(&n, dl, d, du, du2, ipiv, &info);
    }
    return info;
  }

  template <typename Real>
  int gttrs(const char* trans, int n, int nrhs, const Real* dl, const Real* d, const Real* du, const Real* du2, const int* ipiv,
            Real* b, int ldb) {
    int info = 0;
    if constexpr (std::is_same_v<Real, double>) {
      dgttrs_(trans, &n, &nrhs, dl, d, du, du2, ipiv, b, &ldb, &info);
    } else {
      sgttrs_(trans, &n, &nrhs, dl, d, du, du2, ipiv, b, &ldb, &info);
    }
    return info;
  }

  template <typename Real>
  int pttrf(int n, Real* d, Real* e) {
    int info = 0;
    if constexpr (std::is_same_v<Real, double>) {
      dpttrf_(&n, d, e, &info);
    } else {
      spttrf_(&n, d, e, &info);
    }
    return info;
  }

  template <typename Real>
  int pttrs(int n, int nrhs, const Real* d, const Real* e, Real* b, int ldb) {
    int info = 0;
    if constexpr (std::is_same_v<Real, double>) {
      dpttrs_(&n, &nrhs, d, e, b, &ldb, &info);
    } else {
      spttrs_(&n, &nrhs, d, e, b, &ldb, &info);
    }
    return info;
  }

}  // namespace arnoldi_examples::lapackx

#endif
