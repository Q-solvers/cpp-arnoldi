#ifndef ARNOLDI_DETAIL_SORT_HPP
#define ARNOLDI_DETAIL_SORT_HPP

#include <algorithm>
#include <arnoldi/detail/ops.hpp>
#include <cmath>
#include <cstring>

namespace arnoldi::detail {

  template <typename Compare, typename Swap>
  void shell_sort(int n, Compare&& cmp, Swap&& swp) {
    for (int igap = n / 2; igap > 0; igap /= 2)
      for (int i = igap; i < n; i++)
        for (int j = i - igap; j >= 0 && cmp(j, j + igap); j -= igap) swp(j, j + igap);
  }

  template <typename Real>
  void sortc(const char* which, bool apply, int n, Real* xreal, Real* ximag, Real* y) {
    auto swp = [&](int a, int b) {
      std::swap(xreal[a], xreal[b]);
      std::swap(ximag[a], ximag[b]);
      if (apply) std::swap(y[a], y[b]);
    };

    if (std::strcmp(which, "LM") == 0)
      shell_sort(
          n, [&](int a, int b) { return Ops<Real>::lapy2(xreal[a], ximag[a]) > Ops<Real>::lapy2(xreal[b], ximag[b]); }, swp);
    else if (std::strcmp(which, "SM") == 0)
      shell_sort(
          n, [&](int a, int b) { return Ops<Real>::lapy2(xreal[a], ximag[a]) < Ops<Real>::lapy2(xreal[b], ximag[b]); }, swp);
    else if (std::strcmp(which, "LR") == 0)
      shell_sort(
          n, [&](int a, int b) { return xreal[a] > xreal[b]; }, swp);
    else if (std::strcmp(which, "SR") == 0)
      shell_sort(
          n, [&](int a, int b) { return xreal[a] < xreal[b]; }, swp);
    else if (std::strcmp(which, "LI") == 0)
      shell_sort(
          n, [&](int a, int b) { return std::abs(ximag[a]) > std::abs(ximag[b]); }, swp);
    else if (std::strcmp(which, "SI") == 0)
      shell_sort(
          n, [&](int a, int b) { return std::abs(ximag[a]) < std::abs(ximag[b]); }, swp);
  }

  template <typename Real>
  void sortr(const char* which, bool apply, int n, Real* x1, Real* x2) {
    auto swp = [&](int a, int b) {
      std::swap(x1[a], x1[b]);
      if (apply) std::swap(x2[a], x2[b]);
    };

    if (std::strcmp(which, "SA") == 0)
      shell_sort(
          n, [&](int a, int b) { return x1[a] < x1[b]; }, swp);
    else if (std::strcmp(which, "SM") == 0)
      shell_sort(
          n, [&](int a, int b) { return std::abs(x1[a]) < std::abs(x1[b]); }, swp);
    else if (std::strcmp(which, "LA") == 0)
      shell_sort(
          n, [&](int a, int b) { return x1[a] > x1[b]; }, swp);
    else if (std::strcmp(which, "LM") == 0)
      shell_sort(
          n, [&](int a, int b) { return std::abs(x1[a]) > std::abs(x1[b]); }, swp);
  }

  template <typename Real>
  void sesrt(const char* which, bool apply, int n, Real* x, int na, Real* a, int lda) {
    auto swp = [&](int ja, int jb) {
      std::swap(x[ja], x[jb]);
      if (apply) Ops<Real>::swap(na, &a[ja * lda], 1, &a[jb * lda], 1);
    };

    if (std::strcmp(which, "SA") == 0)
      shell_sort(
          n, [&](int a, int b) { return x[a] < x[b]; }, swp);
    else if (std::strcmp(which, "SM") == 0)
      shell_sort(
          n, [&](int a, int b) { return std::abs(x[a]) < std::abs(x[b]); }, swp);
    else if (std::strcmp(which, "LA") == 0)
      shell_sort(
          n, [&](int a, int b) { return x[a] > x[b]; }, swp);
    else if (std::strcmp(which, "LM") == 0)
      shell_sort(
          n, [&](int a, int b) { return std::abs(x[a]) > std::abs(x[b]); }, swp);
  }

}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_SORT_HPP
