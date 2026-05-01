#ifndef ARNOLDI_DETAIL_DEBUG_HPP
#define ARNOLDI_DETAIL_DEBUG_HPP

#include <complex>
#include <cstdio>
#include <type_traits>

namespace arnoldi::detail {

  struct Debug {
    int  logfil = 6;
    int  ndigit = -6;

    int  getv0  = 0;
    int  aupd = 0, aup2 = 0, aitr = 0;
    int  eigh = 0, gets = 0, apps = 0, eupd = 0;

    void ivout(int n, const int* ix, const char* msg) const {
      std::printf("\n %s\n", msg);
      for (int i = 0; i < n; ++i) std::printf(" %12d", ix[i]);
      if (n > 0) std::printf("\n");
    }

    template <typename T>
    void vout(int n, const T* sx, const char* msg) const {
      std::printf("\n %s\n", msg);
      for (int i = 0; i < n; ++i) print_elem(sx[i]);
      if (n > 0) std::printf("\n");
    }

    template <typename T>
    void mout(int m, int n, const T* a, int lda, const char* msg) const {
      std::printf("\n %s\n", msg);
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) print_elem(a[j * lda + i]);
        std::printf("\n");
      }
    }

  private:
    static void print_elem(double v) { std::printf(" %24.16e", v); }
    static void print_elem(float v) { std::printf(" %16.8e", v); }
    static void print_elem(std::complex<double> v) { std::printf(" (%24.16e, %24.16e)", v.real(), v.imag()); }
    static void print_elem(std::complex<float> v) { std::printf(" (%16.8e, %16.8e)", v.real(), v.imag()); }
  };

  inline Debug debug{};

}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_DEBUG_HPP
