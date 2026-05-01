#ifndef ARNOLDI_DETAIL_BLAS_BINDINGS_HPP
#define ARNOLDI_DETAIL_BLAS_BINDINGS_HPP

// Consolidated BLAS / LAPACK C prototypes (Fortran calling convention)
// and value-passing convenience wrappers.
//
// This is the single source of truth for all external linear-algebra symbols
// used by arpack_callback.  Only ops.hpp should include this header; all
// other code accesses BLAS/LAPACK through the Ops<Scalar> dispatch layer.

#include <complex>

// ===========================================================================
//  Raw Fortran prototypes (_-suffixed, pointer-passing)
// ===========================================================================
extern "C" {

// ---- BLAS-1 (double) ------------------------------------------------------
void   dcopy_(const int* n, const double* x, const int* incx, double* y, const int* incy);
void   dswap_(const int* n, double* x, const int* incx, double* y, const int* incy);
void   dscal_(const int* n, const double* a, double* x, const int* incx);
void   daxpy_(const int* n, const double* a, const double* x, const int* incx, double* y, const int* incy);
double ddot_(const int* n, const double* x, const int* incx, const double* y, const int* incy);
double dnrm2_(const int* n, const double* x, const int* incx);

// ---- BLAS-1 (float) -------------------------------------------------------
void   scopy_(const int* n, const float* x, const int* incx, float* y, const int* incy);
void   sswap_(const int* n, float* x, const int* incx, float* y, const int* incy);
void   sscal_(const int* n, const float* a, float* x, const int* incx);
void   saxpy_(const int* n, const float* a, const float* x, const int* incx, float* y, const int* incy);
float  sdot_(const int* n, const float* x, const int* incx, const float* y, const int* incy);
float  snrm2_(const int* n, const float* x, const int* incx);

// ---- BLAS-1 (complex double) ----------------------------------------------
void   zcopy_(const int* n, const std::complex<double>* x, const int* incx, std::complex<double>* y, const int* incy);
void   zswap_(const int* n, std::complex<double>* x, const int* incx, std::complex<double>* y, const int* incy);
void   zscal_(const int* n, const std::complex<double>* a, std::complex<double>* x, const int* incx);
void   zdscal_(const int* n, const double* a, std::complex<double>* x, const int* incx);
void zaxpy_(const int* n, const std::complex<double>* a, const std::complex<double>* x, const int* incx, std::complex<double>* y,
            const int* incy);
double dznrm2_(const int* n, const std::complex<double>* x, const int* incx);
void   cblas_zdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result);

// ---- BLAS-1 (complex float) -----------------------------------------------
void   ccopy_(const int* n, const std::complex<float>* x, const int* incx, std::complex<float>* y, const int* incy);
void   cswap_(const int* n, std::complex<float>* x, const int* incx, std::complex<float>* y, const int* incy);
void   cscal_(const int* n, const std::complex<float>* a, std::complex<float>* x, const int* incx);
void   csscal_(const int* n, const float* a, std::complex<float>* x, const int* incx);
void   caxpy_(const int* n, const std::complex<float>* a, const std::complex<float>* x, const int* incx, std::complex<float>* y,
              const int* incy);
float  scnrm2_(const int* n, const std::complex<float>* x, const int* incx);
void   cblas_cdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result);

// ---- BLAS-2 (gemv, all types) ---------------------------------------------
void dgemv_(const char* trans, const int* m, const int* n, const double* alpha, const double* a, const int* lda, const double* x,
            const int* incx, const double* beta, double* y, const int* incy);
void sgemv_(const char* trans, const int* m, const int* n, const float* alpha, const float* a, const int* lda, const float* x,
            const int* incx, const float* beta, float* y, const int* incy);
void zgemv_(const char* trans, const int* m, const int* n, const std::complex<double>* alpha, const std::complex<double>* a,
            const int* lda, const std::complex<double>* x, const int* incx, const std::complex<double>* beta,
            std::complex<double>* y, const int* incy);
void cgemv_(const char* trans, const int* m, const int* n, const std::complex<float>* alpha, const std::complex<float>* a,
            const int* lda, const std::complex<float>* x, const int* incx, const std::complex<float>* beta,
            std::complex<float>* y, const int* incy);

void dgbmv_(const char* trans, const int* m, const int* n, const int* kl, const int* ku, const double* alpha, const double* a,
            const int* lda, const double* x, const int* incx, const double* beta, double* y, const int* incy);

// ---- BLAS-2 (ger / geru) --------------------------------------------------
void dger_(const int* m, const int* n, const double* alpha, const double* x, const int* incx, const double* y, const int* incy,
           double* a, const int* lda);
void sger_(const int* m, const int* n, const float* alpha, const float* x, const int* incx, const float* y, const int* incy,
           float* a, const int* lda);
void zgeru_(const int* m, const int* n, const std::complex<double>* alpha, const std::complex<double>* x, const int* incx,
            const std::complex<double>* y, const int* incy, std::complex<double>* a, const int* lda);
void cgeru_(const int* m, const int* n, const std::complex<float>* alpha, const std::complex<float>* x, const int* incx,
            const std::complex<float>* y, const int* incy, std::complex<float>* a, const int* lda);

// ---- BLAS-2/3 (trmm, all types) -------------------------------------------
void dtrmm_(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n,
            const double* alpha, const double* a, const int* lda, double* b, const int* ldb);
void strmm_(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n,
            const float* alpha, const float* a, const int* lda, float* b, const int* ldb);
void ztrmm_(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n,
            const std::complex<double>* alpha, const std::complex<double>* a, const int* lda, std::complex<double>* b,
            const int* ldb);
void ctrmm_(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n,
            const std::complex<float>* alpha, const std::complex<float>* a, const int* lda, std::complex<float>* b,
            const int* ldb);

// ---- LAPACK machine constants & norms -------------------------------------
double dlamch_(const char* cmach);
float  slamch_(const char* cmach);
double dlapy2_(const double* x, const double* y);
float  slapy2_(const float* x, const float* y);
void   dlabad_(double* small_val, double* large_val);
void   slabad_(float* small_val, float* large_val);

double dlanhs_(const char* norm, const int* n, const double* a, const int* lda, double* work);
float  slanhs_(const char* norm, const int* n, const float* a, const int* lda, float* work);
double zlanhs_(const char* norm, const int* n, const std::complex<double>* a, const int* lda, double* work);
float  clanhs_(const char* norm, const int* n, const std::complex<float>* a, const int* lda, float* work);

double dlanst_(const char* norm, const int* n, const double* d, const double* e);
float  slanst_(const char* norm, const int* n, const float* d, const float* e);

// ---- LAPACK random vector -------------------------------------------------
void   dlarnv_(const int* idist, int* iseed, const int* n, double* x);
void   slarnv_(const int* idist, int* iseed, const int* n, float* x);
void   zlarnv_(const int* idist, int* iseed, const int* n, std::complex<double>* x);
void   clarnv_(const int* idist, int* iseed, const int* n, std::complex<float>* x);

// ---- LAPACK copy / set / scale --------------------------------------------
void   dlacpy_(const char* uplo, const int* m, const int* n, const double* a, const int* lda, double* b, const int* ldb);
void   slacpy_(const char* uplo, const int* m, const int* n, const float* a, const int* lda, float* b, const int* ldb);
void zlacpy_(const char* uplo, const int* m, const int* n, const std::complex<double>* a, const int* lda, std::complex<double>* b,
             const int* ldb);
void clacpy_(const char* uplo, const int* m, const int* n, const std::complex<float>* a, const int* lda, std::complex<float>* b,
             const int* ldb);

void dlaset_(const char*, const int*, const int*, const double*, const double*, double*, const int*);
void slaset_(const char*, const int*, const int*, const float*, const float*, float*, const int*);
void zlaset_(const char* uplo, const int* m, const int* n, const std::complex<double>* alpha, const std::complex<double>* beta,
             std::complex<double>* a, const int* lda);
void claset_(const char* uplo, const int* m, const int* n, const std::complex<float>* alpha, const std::complex<float>* beta,
             std::complex<float>* a, const int* lda);

void dlascl_(const char*, const int*, const int*, const double*, const double*, const int*, const int*, double*, const int*,
             int*);
void slascl_(const char*, const int*, const int*, const float*, const float*, const int*, const int*, float*, const int*, int*);
void zlascl_(const char*, const int*, const int*, const double*, const double*, const int*, const int*, std::complex<double>*,
             const int*, int*);
void clascl_(const char*, const int*, const int*, const float*, const float*, const int*, const int*, std::complex<float>*,
             const int*, int*);

// ---- LAPACK Householder / Givens rotations --------------------------------
void dlarfg_(const int*, double*, double*, const int*, double*);
void slarfg_(const int*, float*, float*, const int*, float*);

void dlarf_(const char*, const int*, const int*, const double*, const int*, const double*, double*, const int*, double*);
void slarf_(const char*, const int*, const int*, const float*, const int*, const float*, float*, const int*, float*);

void dlartg_(const double*, const double*, double*, double*, double*);
void slartg_(const float*, const float*, float*, float*, float*);
void zlartg_(const std::complex<double>* f, const std::complex<double>* g, double* cs, std::complex<double>* sn,
             std::complex<double>* r);
void clartg_(const std::complex<float>* f, const std::complex<float>* g, float* cs, std::complex<float>* sn,
             std::complex<float>* r);

void dlasr_(const char*, const char*, const char*, const int*, const int*, const double*, const double*, double*, const int*);
void slasr_(const char*, const char*, const char*, const int*, const int*, const float*, const float*, float*, const int*);

// ---- LAPACK symmetric tridiagonal helpers ---------------------------------
void dlae2_(const double*, const double*, const double*, double*, double*);
void slae2_(const float*, const float*, const float*, float*, float*);
void dlaev2_(const double*, const double*, const double*, double*, double*, double*, double*);
void slaev2_(const float*, const float*, const float*, float*, float*, float*, float*);
void dlasrt_(const char*, const int*, double*, int*);
void slasrt_(const char*, const int*, float*, int*);

// ---- LAPACK eigenvalue / Schur (real) -------------------------------------
void dsteqr_(const char* compz, const int* n, double* d, double* e, double* z, const int* ldz, double* work, int* info);
void ssteqr_(const char* compz, const int* n, float* d, float* e, float* z, const int* ldz, float* work, int* info);

void dlahqr_(const int* wantt, const int* wantz, const int* n, const int* ilo, const int* ihi, double* h, const int* ldh,
             double* wr, double* wi, const int* iloz, const int* ihiz, double* z, const int* ldz, int* info);
void slahqr_(const int* wantt, const int* wantz, const int* n, const int* ilo, const int* ihi, float* h, const int* ldh,
             float* wr, float* wi, const int* iloz, const int* ihiz, float* z, const int* ldz, int* info);

void dtrevc_(const char* side, const char* howmny, int* select, const int* n, const double* t, const int* ldt, double* vl,
             const int* ldvl, double* vr, const int* ldvr, const int* mm, int* m, double* work, int* info);
void strevc_(const char* side, const char* howmny, int* select, const int* n, const float* t, const int* ldt, float* vl,
             const int* ldvl, float* vr, const int* ldvr, const int* mm, int* m, float* work, int* info);

void dtrsen_(const char* job, const char* compq, const int* select, const int* n, double* t, const int* ldt, double* q,
             const int* ldq, double* wr, double* wi, int* m, double* s, double* sep, double* work, const int* lwork, int* iwork,
             const int* liwork, int* info);
void strsen_(const char* job, const char* compq, const int* select, const int* n, float* t, const int* ldt, float* q,
             const int* ldq, float* wr, float* wi, int* m, float* s, float* sep, float* work, const int* lwork, int* iwork,
             const int* liwork, int* info);

// ---- LAPACK QR factorisation ----------------------------------------------
void dgeqr2_(const int* m, const int* n, double* a, const int* lda, double* tau, double* work, int* info);
void sgeqr2_(const int* m, const int* n, float* a, const int* lda, float* tau, float* work, int* info);
void zgeqr2_(const int* m, const int* n, std::complex<double>* a, const int* lda, std::complex<double>* tau,
             std::complex<double>* work, int* info);
void cgeqr2_(const int* m, const int* n, std::complex<float>* a, const int* lda, std::complex<float>* tau,
             std::complex<float>* work, int* info);

void dorm2r_(const char* side, const char* trans, const int* m, const int* n, const int* k, const double* a, const int* lda,
             const double* tau, double* c, const int* ldc, double* work, int* info);
void sorm2r_(const char* side, const char* trans, const int* m, const int* n, const int* k, const float* a, const int* lda,
             const float* tau, float* c, const int* ldc, float* work, int* info);
void zunm2r_(const char* side, const char* trans, const int* m, const int* n, const int* k, const std::complex<double>* a,
             const int* lda, const std::complex<double>* tau, std::complex<double>* c, const int* ldc, std::complex<double>* work,
             int* info);
void cunm2r_(const char* side, const char* trans, const int* m, const int* n, const int* k, const std::complex<float>* a,
             const int* lda, const std::complex<float>* tau, std::complex<float>* c, const int* ldc, std::complex<float>* work,
             int* info);

// ---- LAPACK eigenvalue / Schur (complex) ----------------------------------
void zlahqr_(const int* wantt, const int* wantz, const int* n, const int* ilo, const int* ihi, std::complex<double>* h,
             const int* ldh, std::complex<double>* w, const int* iloz, const int* ihiz, std::complex<double>* z, const int* ldz,
             int* info);
void clahqr_(const int* wantt, const int* wantz, const int* n, const int* ilo, const int* ihi, std::complex<float>* h,
             const int* ldh, std::complex<float>* w, const int* iloz, const int* ihiz, std::complex<float>* z, const int* ldz,
             int* info);

void ztrevc_(const char* side, const char* howmny, const int* select, const int* n, std::complex<double>* t, const int* ldt,
             std::complex<double>* vl, const int* ldvl, std::complex<double>* vr, const int* ldvr, const int* mm, int* m,
             std::complex<double>* work, double* rwork, int* info);
void ctrevc_(const char* side, const char* howmny, const int* select, const int* n, std::complex<float>* t, const int* ldt,
             std::complex<float>* vl, const int* ldvl, std::complex<float>* vr, const int* ldvr, const int* mm, int* m,
             std::complex<float>* work, float* rwork, int* info);

void ztrsen_(const char* job, const char* compq, const int* select, const int* n, std::complex<double>* t, const int* ldt,
             std::complex<double>* q, const int* ldq, std::complex<double>* w, int* m, double* s, double* sep,
             std::complex<double>* work, const int* lwork, int* info);
void ctrsen_(const char* job, const char* compq, const int* select, const int* n, std::complex<float>* t, const int* ldt,
             std::complex<float>* q, const int* ldq, std::complex<float>* w, int* m, float* s, float* sep,
             std::complex<float>* work, const int* lwork, int* info);

}  // extern "C"

// ===========================================================================
//  Convenience wrappers (value-passing for scalars)
//  Used only by Ops<Scalar> in ops.hpp.
// ===========================================================================

// ---- double ---------------------------------------------------------------
inline void   dcopy(int n, const double* x, int incx, double* y, int incy) { dcopy_(&n, x, &incx, y, &incy); }
inline void   dswap(int n, double* x, int incx, double* y, int incy) { dswap_(&n, x, &incx, y, &incy); }
inline void   dscal(int n, double a, double* x, int incx) { dscal_(&n, &a, x, &incx); }
inline void   daxpy(int n, double a, const double* x, int incx, double* y, int incy) { daxpy_(&n, &a, x, &incx, y, &incy); }
inline double ddot(int n, const double* x, int incx, const double* y, int incy) { return ddot_(&n, x, &incx, y, &incy); }
inline double dnrm2(int n, const double* x, int incx) { return dnrm2_(&n, x, &incx); }
inline void dgemv(const char* trans, int m, int n, double alpha, const double* a, int lda, const double* x, int incx, double beta,
                  double* y, int incy) {
  dgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}
inline void   dlarnv(int idist, int* iseed, int n, double* x) { dlarnv_(&idist, iseed, &n, x); }
inline double dlamch(const char* cmach) { return dlamch_(cmach); }
inline double dlapy2(double x, double y) { return dlapy2_(&x, &y); }
inline void   dlacpy(const char* uplo, int m, int n, const double* a, int lda, double* b, int ldb) {
    dlacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}
inline void dsteqr(const char* compz, int n, double* d, double* e, double* z, int ldz, double* work, int& info) {
  dsteqr_(compz, &n, d, e, z, &ldz, work, &info);
}
inline void dger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
  dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
inline double dlanhs(const char* norm, int n, const double* a, int lda, double* work) { return dlanhs_(norm, &n, a, &lda, work); }
inline void   dlabad(double& small_val, double& large_val) { dlabad_(&small_val, &large_val); }

// ---- float ----------------------------------------------------------------
inline void   scopy(int n, const float* x, int incx, float* y, int incy) { scopy_(&n, x, &incx, y, &incy); }
inline void   sswap(int n, float* x, int incx, float* y, int incy) { sswap_(&n, x, &incx, y, &incy); }
inline void   sscal(int n, float a, float* x, int incx) { sscal_(&n, &a, x, &incx); }
inline void   saxpy(int n, float a, const float* x, int incx, float* y, int incy) { saxpy_(&n, &a, x, &incx, y, &incy); }
inline float  sdot(int n, const float* x, int incx, const float* y, int incy) { return sdot_(&n, x, &incx, y, &incy); }
inline float  snrm2(int n, const float* x, int incx) { return snrm2_(&n, x, &incx); }
inline void   sgemv(const char* trans, int m, int n, float alpha, const float* a, int lda, const float* x, int incx, float beta,
                    float* y, int incy) {
    sgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}
inline void  slarnv(int idist, int* iseed, int n, float* x) { slarnv_(&idist, iseed, &n, x); }
inline float slamch(const char* cmach) { return slamch_(cmach); }
inline float slapy2(float x, float y) { return slapy2_(&x, &y); }
inline void  slacpy(const char* uplo, int m, int n, const float* a, int lda, float* b, int ldb) {
   slacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}
inline void ssteqr(const char* compz, int n, float* d, float* e, float* z, int ldz, float* work, int& info) {
  ssteqr_(compz, &n, d, e, z, &ldz, work, &info);
}
inline void slabad(float& small_val, float& large_val) { slabad_(&small_val, &large_val); }

// ---- complex double -------------------------------------------------------
inline void zcopy(int n, const std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
  zcopy_(&n, x, &incx, y, &incy);
}
inline void zaxpy(int n, std::complex<double> a, const std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
  zaxpy_(&n, &a, x, &incx, y, &incy);
}
inline void   zscal(int n, std::complex<double> a, std::complex<double>* x, int incx) { zscal_(&n, &a, x, &incx); }
inline void   zdscal(int n, double a, std::complex<double>* x, int incx) { zdscal_(&n, &a, x, &incx); }
inline double dznrm2(int n, const std::complex<double>* x, int incx) { return dznrm2_(&n, x, &incx); }
inline void   zgemv(const char* trans, int m, int n, std::complex<double> alpha, const std::complex<double>* a, int lda,
                    const std::complex<double>* x, int incx, std::complex<double> beta, std::complex<double>* y, int incy) {
    zgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}
inline void zlarnv(int idist, int* iseed, int n, std::complex<double>* x) { zlarnv_(&idist, iseed, &n, x); }
inline void zlacpy(const char* uplo, int m, int n, const std::complex<double>* a, int lda, std::complex<double>* b, int ldb) {
  zlacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}
inline double zlanhs(const char* norm, int n, const std::complex<double>* a, int lda, double* work) {
  return zlanhs_(norm, &n, a, &lda, work);
}
inline void zlaset(const char* uplo, int m, int n, std::complex<double> alpha, std::complex<double> beta, std::complex<double>* a,
                   int lda) {
  zlaset_(uplo, &m, &n, &alpha, &beta, a, &lda);
}
inline void zlartg(std::complex<double> f, std::complex<double> g, double& cs, std::complex<double>& sn,
                   std::complex<double>& r) {
  zlartg_(&f, &g, &cs, &sn, &r);
}
inline void zgeru(int m, int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
                  const std::complex<double>* y, int incy, std::complex<double>* a, int lda) {
  zgeru_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
inline void zgeqr2(int m, int n, std::complex<double>* a, int lda, std::complex<double>* tau, std::complex<double>* work,
                   int& info) {
  zgeqr2_(&m, &n, a, &lda, tau, work, &info);
}
inline void zunm2r(const char* side, const char* trans, int m, int n, int k, const std::complex<double>* a, int lda,
                   const std::complex<double>* tau, std::complex<double>* c, int ldc, std::complex<double>* work, int& info) {
  zunm2r_(side, trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &info);
}
inline void ztrsen(const char* job, const char* compq, const int* select, int n, std::complex<double>* t, int ldt,
                   std::complex<double>* q, int ldq, std::complex<double>* w, int& m, double& s, double& sep,
                   std::complex<double>* work, int lwork, int& info) {
  ztrsen_(job, compq, select, &n, t, &ldt, q, &ldq, w, &m, &s, &sep, work, &lwork, &info);
}
inline void ztrmm(const char* side, const char* uplo, const char* transa, const char* diag, int m, int n,
                  std::complex<double> alpha, const std::complex<double>* a, int lda, std::complex<double>* b, int ldb) {
  ztrmm_(side, uplo, transa, diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

// ---- complex float --------------------------------------------------------
inline void ccopy(int n, const std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
  ccopy_(&n, x, &incx, y, &incy);
}
inline void caxpy(int n, std::complex<float> a, const std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
  caxpy_(&n, &a, x, &incx, y, &incy);
}
inline void  cscal(int n, std::complex<float> a, std::complex<float>* x, int incx) { cscal_(&n, &a, x, &incx); }
inline void  csscal(int n, float a, std::complex<float>* x, int incx) { csscal_(&n, &a, x, &incx); }
inline float scnrm2(int n, const std::complex<float>* x, int incx) { return scnrm2_(&n, x, &incx); }
inline void  cgemv(const char* trans, int m, int n, std::complex<float> alpha, const std::complex<float>* a, int lda,
                   const std::complex<float>* x, int incx, std::complex<float> beta, std::complex<float>* y, int incy) {
   cgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}
inline void clarnv(int idist, int* iseed, int n, std::complex<float>* x) { clarnv_(&idist, iseed, &n, x); }
inline void clacpy(const char* uplo, int m, int n, const std::complex<float>* a, int lda, std::complex<float>* b, int ldb) {
  clacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}
inline float clanhs(const char* norm, int n, const std::complex<float>* a, int lda, float* work) {
  return clanhs_(norm, &n, a, &lda, work);
}
inline void claset(const char* uplo, int m, int n, std::complex<float> alpha, std::complex<float> beta, std::complex<float>* a,
                   int lda) {
  claset_(uplo, &m, &n, &alpha, &beta, a, &lda);
}
inline void clartg(std::complex<float> f, std::complex<float> g, float& cs, std::complex<float>& sn, std::complex<float>& r) {
  clartg_(&f, &g, &cs, &sn, &r);
}
inline void cgeru(int m, int n, std::complex<float> alpha, const std::complex<float>* x, int incx, const std::complex<float>* y,
                  int incy, std::complex<float>* a, int lda) {
  cgeru_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
inline void cgeqr2(int m, int n, std::complex<float>* a, int lda, std::complex<float>* tau, std::complex<float>* work,
                   int& info) {
  cgeqr2_(&m, &n, a, &lda, tau, work, &info);
}
inline void cunm2r(const char* side, const char* trans, int m, int n, int k, const std::complex<float>* a, int lda,
                   const std::complex<float>* tau, std::complex<float>* c, int ldc, std::complex<float>* work, int& info) {
  cunm2r_(side, trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &info);
}
inline void ctrsen(const char* job, const char* compq, const int* select, int n, std::complex<float>* t, int ldt,
                   std::complex<float>* q, int ldq, std::complex<float>* w, int& m, float& s, float& sep,
                   std::complex<float>* work, int lwork, int& info) {
  ctrsen_(job, compq, select, &n, t, &ldt, q, &ldq, w, &m, &s, &sep, work, &lwork, &info);
}
inline void ctrmm(const char* side, const char* uplo, const char* transa, const char* diag, int m, int n,
                  std::complex<float> alpha, const std::complex<float>* a, int lda, std::complex<float>* b, int ldb) {
  ctrmm_(side, uplo, transa, diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

#endif  // ARNOLDI_DETAIL_BLAS_BINDINGS_HPP
