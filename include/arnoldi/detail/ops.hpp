#ifndef ARNOLDI_DETAIL_OPS_HPP
#define ARNOLDI_DETAIL_OPS_HPP

// Unified BLAS/LAPACK dispatch for all four scalar types:
//   float, double, std::complex<float>, std::complex<double>.
//
// Methods that only apply to real types (dot, swap, lapy2, lanhs, and LAPACK
// auxiliary like lahqr_, trevc_, napps helpers) are present but will fail to
// compile if instantiated for complex — which is correct since the Arnoldi
// algorithms only call them on real scalars.

#include <algorithm>
#include <arnoldi/comm.hpp>
#include <arnoldi/detail/blas_bindings.hpp>
#include <arnoldi/detail/stats.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace arnoldi::detail {

  template <typename T>
  struct real_type {
    using type = T;
  };
  template <typename T>
  struct real_type<std::complex<T>> {
    using type = T;
  };
  template <typename T>
  using real_t = typename real_type<T>::type;

  // Non-owning view threaded through the deep algorithmic templates, mirroring
  // the way `const Comm&` is passed today. For CpuBackend it carries no state;
  // device backends specialize this to hold the cuBLAS handle, stream, etc.
  template <typename Backend>
  struct BackendRef {};

  // Default backend: all length-n BLAS calls hit host-side BLAS via blas_bindings.
  // GPU backends (e.g. CudaBackend in arnoldi/cuda.hpp) partially specialize
  // Ops<Scalar, Backend> to route those calls to device libraries instead.
  struct CpuBackend {
    BackendRef<CpuBackend> ref() const noexcept { return {}; }
  };

  // Selects the owning buffer type for Arnoldi's length-n workspace
  // (resid_, v_, workd_) under each Backend, and provides static helpers
  // for transferring data between host pointers and the buffer. Primary
  // template is the CPU path; device backends specialize this (see
  // cuda.hpp). Member access is dependent on Backend, so the right
  // specialization is selected at instantiation.
  template <typename Backend, typename T>
  struct buffer_traits {
    using vector_type = std::vector<T>;

    // Host pointer -> host vector. Trivial std::copy.
    static void copy_from_host(vector_type& dst, const T* src, std::size_t n) {
      std::copy(src, src + n, dst.data());
    }

    // copy_from_device / copy_to_host are not provided for CpuBackend:
    // arnoldi::Arnoldi only invokes them on non-CpuBackend (initial_resid_device
    // is SFINAE'd out, extract_'s else branch is `if constexpr`-discarded).
  };

  template <typename Scalar, typename Backend = CpuBackend>
  struct Ops {
    using Real                       = real_t<Scalar>;
    static constexpr bool is_complex = !std::is_same_v<Scalar, Real>;

    static Real           lamch(const char* cmach) {
                if constexpr (std::is_same_v<Real, double>)
        return dlamch(cmach);
      else
        return slamch(cmach);
    }
    static void labad(Real& a, Real& b) {
      if constexpr (std::is_same_v<Real, double>)
        dlabad(a, b);
      else
        slabad(a, b);
    }
    static Real lapy2(Real x, Real y) {
      if constexpr (std::is_same_v<Real, double>)
        return dlapy2(x, y);
      else
        return slapy2(x, y);
    }

    // Fill a length-n buffer with Scalar(0). CPU: trivial loop. Device backends
    // specialize this to a single asynchronous memset.
    static void zero([[maybe_unused]] BackendRef<Backend> bref, int n, Scalar* x) {
      for (int i = 0; i < n; ++i) x[i] = Scalar(0);
    }

    // Read a single Scalar from a buffer that may be device-resident.
    // CPU: deref. Device backends specialize to cudaMemcpy + sync.
    static Scalar read_scalar([[maybe_unused]] BackendRef<Backend> bref, const Scalar* p) { return *p; }

    // Stage a single Scalar from a (possibly device-resident) buffer into a
    // host slot WITHOUT synchronizing. Callers must Ops::sync() before
    // reading host_dst. CPU: an immediate copy (sync() is then a no-op), so
    // numerics and ordering are bit-identical to a direct read_scalar.
    // Device backends specialize this to an async cudaMemcpy so per-step
    // device->host stalls collapse into one sync per call site batch.
    static void read_scalar_async([[maybe_unused]] BackendRef<Backend> bref, Scalar* host_dst, const Scalar* src) {
      *host_dst = *src;
    }

    // Drain the backend stream so all prior read_scalar_async stages have
    // landed in host memory. CPU: no-op.
    static void sync([[maybe_unused]] BackendRef<Backend> bref) {}

    static void copy([[maybe_unused]] BackendRef<Backend> bref, int n, const Scalar* x, int incx, Scalar* y, int incy) {
      if constexpr (std::is_same_v<Scalar, double>)
        dcopy(n, x, incx, y, incy);
      else if constexpr (std::is_same_v<Scalar, float>)
        scopy(n, x, incx, y, incy);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        zcopy(n, x, incx, y, incy);
      else
        ccopy(n, x, incx, y, incy);
    }
    static void scal([[maybe_unused]] BackendRef<Backend> bref, int n, Scalar a, Scalar* x, int incx) {
      if constexpr (std::is_same_v<Scalar, double>)
        dscal(n, a, x, incx);
      else if constexpr (std::is_same_v<Scalar, float>)
        sscal(n, a, x, incx);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        zscal(n, a, x, incx);
      else
        cscal(n, a, x, incx);
    }
    static void axpy([[maybe_unused]] BackendRef<Backend> bref, int n, Scalar a, const Scalar* x, int incx, Scalar* y, int incy) {
      if constexpr (std::is_same_v<Scalar, double>)
        daxpy(n, a, x, incx, y, incy);
      else if constexpr (std::is_same_v<Scalar, float>)
        saxpy(n, a, x, incx, y, incy);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        zaxpy(n, a, x, incx, y, incy);
      else
        caxpy(n, a, x, incx, y, incy);
    }
    static Real nrm2([[maybe_unused]] BackendRef<Backend> bref, int n, const Scalar* x, int incx) {
      if constexpr (std::is_same_v<Scalar, double>)
        return dnrm2(n, x, incx);
      else if constexpr (std::is_same_v<Scalar, float>)
        return snrm2(n, x, incx);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        return dznrm2(n, x, incx);
      else
        return scnrm2(n, x, incx);
    }
    static void larnv([[maybe_unused]] BackendRef<Backend> bref, int idist, int* iseed, int n, Scalar* x) {
      if constexpr (std::is_same_v<Scalar, double>)
        dlarnv(idist, iseed, n, x);
      else if constexpr (std::is_same_v<Scalar, float>)
        slarnv(idist, iseed, n, x);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        zlarnv(idist, iseed, n, x);
      else
        clarnv(idist, iseed, n, x);
    }

    // Scale complex vector by real factor; for real types collapses to scal().
    static void rscal(BackendRef<Backend> bref, int n, Real a, Scalar* x, int incx) {
      if constexpr (!is_complex)
        scal(bref, n, a, x, incx);
      else if constexpr (std::is_same_v<Real, double>)
        zdscal(n, a, x, incx);
      else
        csscal(n, a, x, incx);
    }
    static void raxpy(BackendRef<Backend> bref, int n, Real a, const Scalar* x, int incx, Scalar* y, int incy) {
      if constexpr (!is_complex)
        axpy(bref, n, a, x, incx, y, incy);
      else
        axpy(bref, n, Scalar(a, 0), x, incx, y, incy);
    }

    static Scalar dot([[maybe_unused]] BackendRef<Backend> bref, int n, const Scalar* x, int incx, const Scalar* y, int incy) {
      if constexpr (std::is_same_v<Scalar, double>)
        return ddot(n, x, incx, y, incy);
      else if constexpr (std::is_same_v<Scalar, float>)
        return sdot(n, x, incx, y, incy);
    }
    static void swap(int n, Scalar* x, int incx, Scalar* y, int incy) {
      if constexpr (std::is_same_v<Scalar, double>)
        dswap(n, x, incx, y, incy);
      else if constexpr (std::is_same_v<Scalar, float>)
        sswap(n, x, incx, y, incy);
    }

    static Scalar dotc([[maybe_unused]] BackendRef<Backend> bref, int n, const Scalar* x, int incx, const Scalar* y, int incy) {
      if constexpr (std::is_same_v<Scalar, double>)
        return ddot(n, x, incx, y, incy);
      else if constexpr (std::is_same_v<Scalar, float>)
        return sdot(n, x, incx, y, incy);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
        Scalar r;
        cblas_zdotc_sub(n, x, incx, y, incy, &r);
        return r;
      } else {
        Scalar r;
        cblas_cdotc_sub(n, x, incx, y, incy, &r);
        return r;
      }
    }
    static Real rdotc(BackendRef<Backend> bref, int n, const Scalar* x, int incx, const Scalar* y, int incy) {
      return std::real(dotc(bref, n, x, incx, y, incy));
    }

    static const char* herm_trans() {
      if constexpr (is_complex)
        return "C";
      else
        return "T";
    }

    static void gemv([[maybe_unused]] BackendRef<Backend> bref, const char* trans, int m, int n, Scalar alpha, const Scalar* a,
                     int lda, const Scalar* x, int incx, Scalar beta, Scalar* y, int incy) {
      if constexpr (std::is_same_v<Scalar, double>)
        dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
      else if constexpr (std::is_same_v<Scalar, float>)
        sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
      else
        cgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void gemm([[maybe_unused]] BackendRef<Backend> bref, const char* transa, const char* transb, int m, int n, int k,
                     Scalar alpha, const Scalar* a, int lda, const Scalar* b, int ldb, Scalar beta, Scalar* c, int ldc) {
      if constexpr (std::is_same_v<Scalar, double>)
        dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      else if constexpr (std::is_same_v<Scalar, float>)
        sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      else
        cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    // gemv with a real x-vector applied to a possibly-complex matrix.
    static void gemv_rv(BackendRef<Backend> bref, const char* trans, int m, int ncols, Real alpha, const Scalar* a, int lda,
                        const Real* x, int incx, Real beta, Scalar* y, int incy) {
      if constexpr (!is_complex) {
        gemv(bref, trans, m, ncols, alpha, a, lda, x, incx, beta, y, incy);
      } else {
        int                 xlen = (*trans == 'N' || *trans == 'n') ? ncols : m;
        std::vector<Scalar> cx(xlen);
        for (int i = 0; i < xlen; i++) cx[i] = Scalar(x[i * incx], 0);
        gemv(bref, trans, m, ncols, Scalar(alpha, 0), a, lda, cx.data(), 1, Scalar(beta, 0), y, incy);
      }
    }

    // -----------------------------------------------------------------------
    // No-BackendRef overloads — for host-only call sites (e.g. eig.hpp's
    // seigt / neigh on workl/h/q, and other small host-side BLAS-1/2 calls).
    // These forward to the BackendRef-taking version with an empty
    // BackendRef<Backend>{}; under CudaBackend the partial specialization
    // does NOT provide these, so an accidental device-side host-style call
    // becomes a compile error.
    static void   zero(int n, Scalar* x) { zero(BackendRef<Backend>{}, n, x); }
    static Scalar read_scalar(const Scalar* p) { return read_scalar(BackendRef<Backend>{}, p); }
    static void copy(int n, const Scalar* x, int incx, Scalar* y, int incy) { copy(BackendRef<Backend>{}, n, x, incx, y, incy); }
    static void scal(int n, Scalar a, Scalar* x, int incx) { scal(BackendRef<Backend>{}, n, a, x, incx); }
    static void axpy(int n, Scalar a, const Scalar* x, int incx, Scalar* y, int incy) {
      axpy(BackendRef<Backend>{}, n, a, x, incx, y, incy);
    }
    static Real nrm2(int n, const Scalar* x, int incx) { return nrm2(BackendRef<Backend>{}, n, x, incx); }
    static void larnv(int idist, int* iseed, int n, Scalar* x) { larnv(BackendRef<Backend>{}, idist, iseed, n, x); }
    static void rscal(int n, Real a, Scalar* x, int incx) { rscal(BackendRef<Backend>{}, n, a, x, incx); }
    static void raxpy(int n, Real a, const Scalar* x, int incx, Scalar* y, int incy) {
      raxpy(BackendRef<Backend>{}, n, a, x, incx, y, incy);
    }
    static Scalar dot(int n, const Scalar* x, int incx, const Scalar* y, int incy) {
      return dot(BackendRef<Backend>{}, n, x, incx, y, incy);
    }
    static Scalar dotc(int n, const Scalar* x, int incx, const Scalar* y, int incy) {
      return dotc(BackendRef<Backend>{}, n, x, incx, y, incy);
    }
    static Real rdotc(int n, const Scalar* x, int incx, const Scalar* y, int incy) {
      return rdotc(BackendRef<Backend>{}, n, x, incx, y, incy);
    }
    static void gemv(const char* trans, int m, int n, Scalar alpha, const Scalar* a, int lda, const Scalar* x, int incx,
                     Scalar beta, Scalar* y, int incy) {
      gemv(BackendRef<Backend>{}, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
    static void gemm(const char* transa, const char* transb, int m, int n, int k, Scalar alpha, const Scalar* a, int lda,
                     const Scalar* b, int ldb, Scalar beta, Scalar* c, int ldc) {
      gemm(BackendRef<Backend>{}, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    static void gemv_rv(const char* trans, int m, int ncols, Real alpha, const Scalar* a, int lda, const Real* x, int incx,
                        Real beta, Scalar* y, int incy) {
      gemv_rv(BackendRef<Backend>{}, trans, m, ncols, alpha, a, lda, x, incx, beta, y, incy);
    }
    static void lascl_(const char* type, const int* kl, const int* ku, const Real* cfrom, const Real* cto, const int* m,
                       const int* n, Scalar* a, const int* lda, int* info) {
      lascl_(BackendRef<Backend>{}, type, kl, ku, cfrom, cto, m, n, a, lda, info);
    }

    static void ger(int m, int n, Scalar alpha, const Scalar* x, int incx, const Scalar* y, int incy, Scalar* a, int lda) {
      if constexpr (std::is_same_v<Scalar, double>)
        dger(m, n, alpha, x, incx, y, incy, a, lda);
      else if constexpr (std::is_same_v<Scalar, float>)
        sger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
    }
    static Real lanhs(const char* norm, int n, const Scalar* a, int lda, Scalar* work) {
      if constexpr (std::is_same_v<Scalar, double>)
        return dlanhs(norm, n, a, lda, work);
      else if constexpr (std::is_same_v<Scalar, float>)
        return ::slanhs_(norm, &n, a, &lda, work);
    }

    static void lacpy(const char* uplo, int m, int n, const Scalar* a, int lda, Scalar* b, int ldb) {
      if constexpr (std::is_same_v<Scalar, double>)
        dlacpy(uplo, m, n, a, lda, b, ldb);
      else if constexpr (std::is_same_v<Scalar, float>)
        slacpy(uplo, m, n, a, lda, b, ldb);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        zlacpy(uplo, m, n, a, lda, b, ldb);
      else
        clacpy(uplo, m, n, a, lda, b, ldb);
    }
    static void lascl_([[maybe_unused]] BackendRef<Backend> bref, const char* type, const int* kl, const int* ku, const Real* cfrom,
                       const Real* cto, const int* m, const int* n, Scalar* a, const int* lda, int* info) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlascl_(type, kl, ku, cfrom, cto, m, n, a, lda, info);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slascl_(type, kl, ku, cfrom, cto, m, n, a, lda, info);
      else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
        ::zlascl_(type, kl, ku, cfrom, cto, m, n, a, lda, info);
      else
        ::clascl_(type, kl, ku, cfrom, cto, m, n, a, lda, info);
    }

    static void steqr(const char* compz, int n, Real* d, Real* e, Real* z, int ldz, Real* work, int& info) {
      if constexpr (std::is_same_v<Real, double>)
        dsteqr(compz, n, d, e, z, ldz, work, info);
      else
        ssteqr(compz, n, d, e, z, ldz, work, info);
    }

    static void laset_(const char* uplo, const int* m, const int* n, const Scalar* alpha, const Scalar* beta, Scalar* a,
                       const int* lda) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlaset_(uplo, m, n, alpha, beta, a, lda);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slaset_(uplo, m, n, alpha, beta, a, lda);
    }
    static void larfg_(const int* n, Scalar* alpha, Scalar* x, const int* incx, Scalar* tau) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlarfg_(n, alpha, x, incx, tau);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slarfg_(n, alpha, x, incx, tau);
    }
    static void larf_(const char* side, const int* m, const int* n, const Scalar* v, const int* incv, const Scalar* tau,
                      Scalar* c, const int* ldc, Scalar* work) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlarf_(side, m, n, v, incv, tau, c, ldc, work);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slarf_(side, m, n, v, incv, tau, c, ldc, work);
    }
    static void lartg_(const Scalar* f, const Scalar* g, Scalar* cs, Scalar* sn, Scalar* r) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlartg_(f, g, cs, sn, r);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slartg_(f, g, cs, sn, r);
    }
    static void lahqr_(const int* wantt, const int* wantz, const int* n, const int* ilo, const int* ihi, Scalar* h,
                       const int* ldh, Scalar* wr, Scalar* wi, const int* iloz, const int* ihiz, Scalar* z, const int* ldz,
                       int* info) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlahqr_(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slahqr_(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
    }
    static void trevc_(const char* side, const char* howmny, int* select, const int* n, const Scalar* t, const int* ldt,
                       Scalar* vl, const int* ldvl, Scalar* vr, const int* ldvr, const int* mm, int* m, Scalar* work, int* info) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dtrevc_(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::strevc_(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info);
    }
    static void geqr2_(const int* m, const int* n, Scalar* a, const int* lda, Scalar* tau, Scalar* work, int* info) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dgeqr2_(m, n, a, lda, tau, work, info);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::sgeqr2_(m, n, a, lda, tau, work, info);
    }
    static void orm2r_(const char* side, const char* trans, const int* m, const int* n, const int* k, const Scalar* a,
                       const int* lda, const Scalar* tau, Scalar* c, const int* ldc, Scalar* work, int* info) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dorm2r_(side, trans, m, n, k, a, lda, tau, c, ldc, work, info);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::sorm2r_(side, trans, m, n, k, a, lda, tau, c, ldc, work, info);
    }
    static void trsen_(const char* job, const char* compq, const int* select, const int* n, Scalar* t, const int* ldt, Scalar* q,
                       const int* ldq, Scalar* wr, Scalar* wi, int* m, Scalar* s, Scalar* sep, Scalar* work, const int* lwork,
                       int* iwork, const int* liwork, int* info) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dtrsen_(job, compq, select, n, t, ldt, q, ldq, wr, wi, m, s, sep, work, lwork, iwork, liwork, info);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::strsen_(job, compq, select, n, t, ldt, q, ldq, wr, wi, m, s, sep, work, lwork, iwork, liwork, info);
    }
    static void trmm_(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n,
                      const Scalar* alpha, const Scalar* a, const int* lda, Scalar* b, const int* ldb) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dtrmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::strmm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
    static void lae2_(const Scalar* a, const Scalar* b, const Scalar* c, Scalar* rt1, Scalar* rt2) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlae2_(a, b, c, rt1, rt2);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slae2_(a, b, c, rt1, rt2);
    }
    static void laev2_(const Scalar* a, const Scalar* b, const Scalar* c, Scalar* rt1, Scalar* rt2, Scalar* cs, Scalar* sn) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlaev2_(a, b, c, rt1, rt2, cs, sn);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slaev2_(a, b, c, rt1, rt2, cs, sn);
    }
    static void lasr_(const char* side, const char* pivot, const char* direct, const int* m, const int* n, const Scalar* c,
                      const Scalar* s, Scalar* a, const int* lda) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlasr_(side, pivot, direct, m, n, c, s, a, lda);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slasr_(side, pivot, direct, m, n, c, s, a, lda);
    }
    static void lasrt_(const char* id, const int* n, Scalar* d, int* info) {
      if constexpr (std::is_same_v<Scalar, double>)
        ::dlasrt_(id, n, d, info);
      else if constexpr (std::is_same_v<Scalar, float>)
        ::slasrt_(id, n, d, info);
    }
    static Real lanst_(const char* norm, const int* n, const Scalar* d, const Scalar* e) {
      if constexpr (std::is_same_v<Scalar, double>)
        return ::dlanst_(norm, n, d, e);
      else if constexpr (std::is_same_v<Scalar, float>)
        return ::slanst_(norm, n, d, e);
    }
  };

  // Distributed reduction helpers (PARPACK-style).
  // For SerialComm these compile to bare BLAS calls with zero overhead.
  // Backend defaults to CpuBackend so existing pdot<Real>(...) call sites keep
  // compiling; algorithmic code that lives under a Backend template forwards
  // it explicitly: pdot<Real, Backend>(bref, ...).
  template <typename Real, typename Backend = CpuBackend, typename Comm>
  Real pdot(BackendRef<Backend> bref, const Comm& c, int n, const Real* x, int incx, const Real* y, int incy) {
    return c.allreduce_sum(Ops<Real, Backend>::dot(bref, n, x, incx, y, incy));
  }
  template <typename Scalar, typename Backend = CpuBackend, typename Comm>
  real_t<Scalar> prdotc(BackendRef<Backend> bref, const Comm& c, int n, const Scalar* x, int incx, const Scalar* y, int incy) {
    return c.allreduce_sum(Ops<Scalar, Backend>::rdotc(bref, n, x, incx, y, incy));
  }
  template <typename Real, typename Backend = CpuBackend, typename Comm>
  Real pnrm2_real(BackendRef<Backend> bref, const Comm& c, int n, const Real* x, int incx) {
    Real s = c.allreduce_sum(Ops<Real, Backend>::dot(bref, n, x, incx, x, incx));
    return std::sqrt(std::abs(s));
  }
  template <typename Scalar, typename Backend = CpuBackend, typename Comm>
  real_t<Scalar> pnrm2(BackendRef<Backend> bref, const Comm& c, int n, const Scalar* x, int incx) {
    auto s = c.allreduce_sum(Ops<Scalar, Backend>::rdotc(bref, n, x, incx, x, incx));
    return std::sqrt(std::abs(s));
  }

  // B-norm helper: compute rnorm = B-norm(resid) using workd as scratch.
  // For bmat='G': bop(resid, workd[0..]), rnorm = sqrt(|resid . workd|).
  // For bmat='I': rnorm = nrm2(resid).
  template <typename Scalar, typename Backend = CpuBackend, typename BOP, typename Comm>
  real_t<Scalar> bnorm(BackendRef<Backend> bref, const char bmat, int n, Scalar* resid, Scalar* workd_lo, Scalar* workd_hi,
                       BOP&& bop, const Comm& comm) {
    using Real = real_t<Scalar>;
    Real val;
    if (bmat == 'G') {
      stats.nbx++;
      Ops<Scalar, Backend>::copy(bref, n, resid, 1, workd_hi, 1);
      bop(workd_hi, workd_lo);
      val = prdotc<Scalar, Backend>(bref, comm, n, resid, 1, workd_lo, 1);
      val = std::sqrt(std::abs(val));
    } else {
      Ops<Scalar, Backend>::copy(bref, n, resid, 1, workd_lo, 1);
      val = pnrm2<Scalar, Backend>(bref, comm, n, resid, 1);
    }
    return val;
  }

  // ---------------------------------------------------------------------------
  // No-BackendRef overloads for the reduction helpers — defaults to CpuBackend
  // for the historical public API (tests, examples).
  template <typename Real, typename Comm>
  Real pdot(const Comm& c, int n, const Real* x, int incx, const Real* y, int incy) {
    return pdot<Real, CpuBackend>(BackendRef<CpuBackend>{}, c, n, x, incx, y, incy);
  }
  template <typename Scalar, typename Comm>
  real_t<Scalar> prdotc(const Comm& c, int n, const Scalar* x, int incx, const Scalar* y, int incy) {
    return prdotc<Scalar, CpuBackend>(BackendRef<CpuBackend>{}, c, n, x, incx, y, incy);
  }
  template <typename Real, typename Comm>
  Real pnrm2_real(const Comm& c, int n, const Real* x, int incx) {
    return pnrm2_real<Real, CpuBackend>(BackendRef<CpuBackend>{}, c, n, x, incx);
  }
  template <typename Scalar, typename Comm>
  real_t<Scalar> pnrm2(const Comm& c, int n, const Scalar* x, int incx) {
    return pnrm2<Scalar, CpuBackend>(BackendRef<CpuBackend>{}, c, n, x, incx);
  }
  template <typename Scalar, typename BOP, typename Comm>
  real_t<Scalar> bnorm(const char bmat, int n, Scalar* resid, Scalar* workd_lo, Scalar* workd_hi, BOP&& bop, const Comm& comm) {
    return bnorm<Scalar, CpuBackend>(BackendRef<CpuBackend>{}, bmat, n, resid, workd_lo, workd_hi, std::forward<BOP>(bop), comm);
  }

  // ---------------------------------------------------------------------------
  // Backend-aware extension points. Primary versions are CPU; the CUDA
  // backend overrides them via additional overloads declared in cuda.hpp.
  // Algorithmic templates call these by name; ADL + overload resolution
  // picks the right one based on the BackendRef argument.
  // ---------------------------------------------------------------------------

  // In-place allreduce on a length-`len` buffer. Under CpuBackend the buffer
  // is host-resident, so we forward to comm.allreduce_sum directly. The CUDA
  // backend provides an overload that stages D->H, allreduces on a host
  // scratch buffer, then copies H->D. Length is always small (<= ncv) — the
  // staging cost is irrelevant compared to the matvec.
  template <typename Backend, typename Scalar, typename Comm>
  void backend_allreduce(BackendRef<Backend>, const Comm& comm, Scalar* buf, int len) {
    comm.allreduce_sum(buf, len);
  }


}  // namespace arnoldi::detail

#endif  // ARNOLDI_DETAIL_OPS_HPP
