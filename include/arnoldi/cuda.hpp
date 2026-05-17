#ifndef ARNOLDI_CUDA_HPP
#define ARNOLDI_CUDA_HPP

// CUDA + cuBLAS backend for the symmetric / Hermitian Arnoldi path.
//
// Gated behind ARNOLDI_USE_CUDA. When the macro is defined, this header
// provides:
//   - arnoldi::detail::CudaBackend         (RAII over cuBLAS handle + stream)
//   - arnoldi::detail::BackendRef<CudaBackend> partial spec (handle + stream)
//   - arnoldi::detail::Ops<Scalar, CudaBackend> partial spec (cuBLAS dispatch)
//
// Callback contract under CudaBackend:
//
//   The user matvec callback `op(const Scalar* x, Scalar* y)` receives
//   DEVICE pointers. The matvec runs on the stream returned by
//   backend.stream(); if the user launches on a different stream they
//   must synchronise it before returning, otherwise the subsequent
//   cuBLAS calls will race against the user kernel.
//
// Eigenvalues: extracted by seupd into host arrays (Real eigenvalues, plus
// optionally Scalar eigenvectors after a final cublasGEMM). The Arnoldi
// class transfers the result to host std::vector on eigenpairs() return.
//
// Pointer-mode policy: CUBLAS_POINTER_MODE_HOST. dotc/nrm2 return a host
// scalar synchronously, matching CPU semantics exactly.
//
// Supported scalar types: double, float, std::complex<double>,
// std::complex<float>. Sym + real, Herm + complex.

#ifdef ARNOLDI_USE_CUDA

#include <arnoldi/detail/ops.hpp>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace arnoldi {

  namespace detail {

    // Internal error-check helpers. Keep them small and noisy; on failure
    // throw so the user-facing exception carries a meaningful message.
    inline void cuda_check(cudaError_t e, const char* what) {
      if (e != cudaSuccess) {
        throw std::runtime_error(std::string("arnoldi/cuda: ") + what + ": " + cudaGetErrorString(e));
      }
    }
    inline void cublas_check(cublasStatus_t s, const char* what) {
      if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("arnoldi/cuda: ") + what + ": cublas error " + std::to_string(static_cast<int>(s)));
      }
    }

    inline cublasOperation_t cublas_op(const char* s) {
      switch (s[0]) {
        case 'N': case 'n': return CUBLAS_OP_N;
        case 'T': case 't': return CUBLAS_OP_T;
        case 'C': case 'c': return CUBLAS_OP_C;
      }
      return CUBLAS_OP_N;
    }

  }  // namespace detail

  // RAII over a cuBLAS handle bound to a stream. The Arnoldi class stores
  // one of these by value; ref() returns a trivially-copyable view that
  // gets threaded through the algorithmic templates.
  //
  // Construction modes:
  //   CudaBackend()                       — creates own handle + stream.
  //   CudaBackend(cudaStream_t s)         — uses caller's stream; creates own
  //                                          handle bound to that stream.
  //
  // Non-copyable, movable. Default ctor owns its stream and destroys it.
  class CudaBackend {
  public:
    CudaBackend() : owns_stream_(true) {
      detail::cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
      detail::cublas_check(cublasCreate(&handle_), "cublasCreate");
      detail::cublas_check(cublasSetStream(handle_, stream_), "cublasSetStream");
      detail::cublas_check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(HOST)");
    }
    explicit CudaBackend(cudaStream_t s) : stream_(s), owns_stream_(false) {
      detail::cublas_check(cublasCreate(&handle_), "cublasCreate");
      detail::cublas_check(cublasSetStream(handle_, stream_), "cublasSetStream");
      detail::cublas_check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(HOST)");
    }

    CudaBackend(const CudaBackend&)            = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    CudaBackend(CudaBackend&& other) noexcept :
        handle_(other.handle_), stream_(other.stream_), owns_stream_(other.owns_stream_) {
      other.handle_       = nullptr;
      other.stream_       = nullptr;
      other.owns_stream_  = false;
    }
    CudaBackend& operator=(CudaBackend&& other) noexcept {
      if (this != &other) {
        release_();
        handle_            = other.handle_;
        stream_            = other.stream_;
        owns_stream_       = other.owns_stream_;
        other.handle_      = nullptr;
        other.stream_      = nullptr;
        other.owns_stream_ = false;
      }
      return *this;
    }

    ~CudaBackend() { release_(); }

    cublasHandle_t handle() const noexcept { return handle_; }
    cudaStream_t   stream() const noexcept { return stream_; }

    detail::BackendRef<CudaBackend> ref() const noexcept;

  private:
    void release_() noexcept {
      if (handle_) cublasDestroy(handle_);
      if (owns_stream_ && stream_) cudaStreamDestroy(stream_);
      handle_      = nullptr;
      stream_      = nullptr;
      owns_stream_ = false;
    }

    cublasHandle_t handle_      = nullptr;
    cudaStream_t   stream_      = nullptr;
    bool           owns_stream_ = false;
  };

  namespace detail {

    // Non-owning view threaded through the algorithmic templates.
    template <>
    struct BackendRef<CudaBackend> {
      cublasHandle_t handle = nullptr;
      cudaStream_t   stream = nullptr;
    };

    // Minimal RAII vector over device memory. Mirrors the small slice of
    // std::vector<T> that the Arnoldi class actually uses: default ctor,
    // .assign(n, T{}) (zero-fill via cudaMemset), .data(), .size(), and a
    // destructor that frees. Non-copyable, movable.
    //
    // IEEE-754 zero is all-bits-zero for both real and complex (each
    // component zero), so a single cudaMemset suffices for the zero-fill
    // initialization that Arnoldi's ctor performs.
    template <typename T>
    class device_vector {
    public:
      device_vector() noexcept = default;

      device_vector(const device_vector&)            = delete;
      device_vector& operator=(const device_vector&) = delete;

      device_vector(device_vector&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
      }
      device_vector& operator=(device_vector&& other) noexcept {
        if (this != &other) {
          release_();
          data_       = other.data_;
          size_       = other.size_;
          other.data_ = nullptr;
          other.size_ = 0;
        }
        return *this;
      }

      ~device_vector() { release_(); }

      // Allocate n elements and fill with `val`. Currently only val == T{}
      // (zero) is needed by Arnoldi; we'd add a host-staged fill if we ever
      // needed a non-zero initial value.
      void assign(std::size_t n, const T& val) {
        resize_(n);
        if (n == 0) return;
        if (val == T{}) {
          cuda_check(cudaMemset(data_, 0, n * sizeof(T)), "device_vector::assign(zero) cudaMemset");
        } else {
          std::vector<T> host(n, val);
          cuda_check(cudaMemcpy(data_, host.data(), n * sizeof(T), cudaMemcpyHostToDevice),
                     "device_vector::assign(fill) cudaMemcpy");
        }
      }

      T*          data() noexcept { return data_; }
      const T*    data() const noexcept { return data_; }
      std::size_t size() const noexcept { return size_; }

    private:
      void resize_(std::size_t n) {
        if (size_ == n) return;
        release_();
        if (n > 0) {
          void* p = nullptr;
          cuda_check(cudaMalloc(&p, n * sizeof(T)), "device_vector::resize_ cudaMalloc");
          data_ = static_cast<T*>(p);
          size_ = n;
        }
      }
      void release_() noexcept {
        if (data_) cudaFree(data_);
        data_ = nullptr;
        size_ = 0;
      }

      T*          data_ = nullptr;
      std::size_t size_ = 0;
    };

    // Buffer-type trait specialization: under CudaBackend the Arnoldi
    // workspace lives in device_vector. Static helpers shuttle data
    // between host pointers and the device buffer for initial_resid,
    // initial_resid_device, and extract_'s eigenvector copy-down.
    template <typename T>
    struct buffer_traits<CudaBackend, T> {
      using vector_type = device_vector<T>;

      // Host pointer -> device buffer (initial_resid upload).
      static void copy_from_host(vector_type& dst, const T* src, std::size_t n) {
        cuda_check(cudaMemcpy(dst.data(), src, n * sizeof(T), cudaMemcpyHostToDevice), "buffer_traits::copy_from_host");
      }

      // Device pointer -> device buffer (initial_resid_device pass-through).
      static void copy_from_device(vector_type& dst, const T* src, std::size_t n) {
        cuda_check(cudaMemcpy(dst.data(), src, n * sizeof(T), cudaMemcpyDeviceToDevice), "buffer_traits::copy_from_device");
      }

      // Device buffer -> host pointer (extract_ eigenvector copy-down).
      static void copy_to_host(T* host_dst, const vector_type& src, std::size_t n) {
        cuda_check(cudaMemcpy(host_dst, src.data(), n * sizeof(T), cudaMemcpyDeviceToHost), "buffer_traits::copy_to_host");
      }
    };

    // Overloads for the backend-aware extension points (declared in ops.hpp).

    // Length-`len` allreduce on a device buffer: stage D->H, allreduce on a
    // host scratch buffer, copy H->D, sync. Always small (<= ncv).
    template <typename Scalar, typename Comm>
    void backend_allreduce(BackendRef<CudaBackend> bref, const Comm& comm, Scalar* buf, int len) {
      // SerialComm::allreduce_sum is an identity no-op. Staging the buffer
      // D->H->D and synchronizing twice purely to feed it is wasted work on
      // the Lanczos hot path — skip the round-trip entirely. The buffer is
      // already correct in device memory.
      if constexpr (std::is_same_v<Comm, ::arnoldi::SerialComm>) {
        (void)bref;
        (void)comm;
        (void)buf;
        (void)len;
        return;
      } else {
        std::vector<Scalar> host_buf(static_cast<std::size_t>(len));
      cuda_check(cudaMemcpyAsync(host_buf.data(), buf, static_cast<std::size_t>(len) * sizeof(Scalar), cudaMemcpyDeviceToHost,
                                 bref.stream),
                 "backend_allreduce D2H");
      cuda_check(cudaStreamSynchronize(bref.stream), "backend_allreduce sync D2H");
      comm.allreduce_sum(host_buf.data(), len);
      cuda_check(cudaMemcpyAsync(buf, host_buf.data(), static_cast<std::size_t>(len) * sizeof(Scalar), cudaMemcpyHostToDevice,
                                 bref.stream),
                 "backend_allreduce H2D");
      cuda_check(cudaStreamSynchronize(bref.stream), "backend_allreduce sync H2D");
      }
    }

  }  // namespace detail

  inline detail::BackendRef<CudaBackend> CudaBackend::ref() const noexcept {
    return detail::BackendRef<CudaBackend>{handle_, stream_};
  }

  namespace detail {

    // Ops<Scalar, CudaBackend> partial specialisation.
    //
    // Provides the length-n dispatch surface for the Sym / Herm path:
    //   copy/scal/rscal/axpy/raxpy/nrm2/dotc/rdotc/gemv/gemm/gemv_rv/larnv/
    //   lascl_/zero/read_scalar
    //
    // Does NOT provide the host-only LAPACK aux (lamch, lapy2, steqr,
    // laset_, larfg_, lartg_, lahqr_, trevc_, geqr2_, trsen_, trmm_, ger,
    // lacpy, lanhs, lae2_, laev2_, lasr_, lasrt_, lanst_, swap). Those run
    // on host-resident workl/h/q via the primary CPU Ops template, which
    // is reached by Ops<Real>::method(...) call sites (default CpuBackend).
    template <typename Scalar>
    struct Ops<Scalar, CudaBackend> {
      using Real                       = real_t<Scalar>;
      static constexpr bool is_complex = !std::is_same_v<Scalar, Real>;
      using BRef                       = BackendRef<CudaBackend>;

      // ---- fills / scalar reads ---------------------------------------------
      static void zero(BRef bref, int n, Scalar* x) {
        // IEEE-754 zero is all-bits-zero for both real and complex (each
        // component zero). cudaMemsetAsync is async on the stream.
        cuda_check(cudaMemsetAsync(x, 0, static_cast<std::size_t>(n) * sizeof(Scalar), bref.stream), "cudaMemsetAsync(zero)");
      }

      static Scalar read_scalar(BRef bref, const Scalar* p) {
        Scalar host_val{};
        cuda_check(cudaMemcpyAsync(&host_val, p, sizeof(Scalar), cudaMemcpyDeviceToHost, bref.stream),
                   "cudaMemcpyAsync(read_scalar)");
        cuda_check(cudaStreamSynchronize(bref.stream), "cudaStreamSynchronize(read_scalar)");
        return host_val;
      }

      // Async single-element D->H stage (no per-call sync). The caller batches
      // many of these and issues one sync() before reading host memory,
      // collapsing what would be one stream stall per Lanczos step into one
      // stall per saitr call.
      static void read_scalar_async(BRef bref, Scalar* host_dst, const Scalar* src) {
        cuda_check(cudaMemcpyAsync(host_dst, src, sizeof(Scalar), cudaMemcpyDeviceToHost, bref.stream),
                   "cudaMemcpyAsync(read_scalar_async)");
      }

      static void sync(BRef bref) { cuda_check(cudaStreamSynchronize(bref.stream), "cudaStreamSynchronize(sync)"); }

      // ---- BLAS-1 -----------------------------------------------------------
      static void copy(BRef bref, int n, const Scalar* x, int incx, Scalar* y, int incy) {
        if constexpr (std::is_same_v<Scalar, double>)
          cublas_check(cublasDcopy(bref.handle, n, x, incx, y, incy), "cublasDcopy");
        else if constexpr (std::is_same_v<Scalar, float>)
          cublas_check(cublasScopy(bref.handle, n, x, incx, y, incy), "cublasScopy");
        else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
          cublas_check(cublasZcopy(bref.handle, n, reinterpret_cast<const cuDoubleComplex*>(x), incx,
                                   reinterpret_cast<cuDoubleComplex*>(y), incy),
                       "cublasZcopy");
        else
          cublas_check(cublasCcopy(bref.handle, n, reinterpret_cast<const cuComplex*>(x), incx,
                                   reinterpret_cast<cuComplex*>(y), incy),
                       "cublasCcopy");
      }

      static void scal(BRef bref, int n, Scalar a, Scalar* x, int incx) {
        if constexpr (std::is_same_v<Scalar, double>)
          cublas_check(cublasDscal(bref.handle, n, &a, x, incx), "cublasDscal");
        else if constexpr (std::is_same_v<Scalar, float>)
          cublas_check(cublasSscal(bref.handle, n, &a, x, incx), "cublasSscal");
        else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
          cublas_check(cublasZscal(bref.handle, n, reinterpret_cast<const cuDoubleComplex*>(&a),
                                   reinterpret_cast<cuDoubleComplex*>(x), incx),
                       "cublasZscal");
        else
          cublas_check(cublasCscal(bref.handle, n, reinterpret_cast<const cuComplex*>(&a),
                                   reinterpret_cast<cuComplex*>(x), incx),
                       "cublasCscal");
      }

      static void rscal(BRef bref, int n, Real a, Scalar* x, int incx) {
        if constexpr (!is_complex) {
          scal(bref, n, a, x, incx);
        } else if constexpr (std::is_same_v<Real, double>) {
          cublas_check(cublasZdscal(bref.handle, n, &a, reinterpret_cast<cuDoubleComplex*>(x), incx), "cublasZdscal");
        } else {
          cublas_check(cublasCsscal(bref.handle, n, &a, reinterpret_cast<cuComplex*>(x), incx), "cublasCsscal");
        }
      }

      static void axpy(BRef bref, int n, Scalar a, const Scalar* x, int incx, Scalar* y, int incy) {
        if constexpr (std::is_same_v<Scalar, double>)
          cublas_check(cublasDaxpy(bref.handle, n, &a, x, incx, y, incy), "cublasDaxpy");
        else if constexpr (std::is_same_v<Scalar, float>)
          cublas_check(cublasSaxpy(bref.handle, n, &a, x, incx, y, incy), "cublasSaxpy");
        else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
          cublas_check(cublasZaxpy(bref.handle, n, reinterpret_cast<const cuDoubleComplex*>(&a),
                                   reinterpret_cast<const cuDoubleComplex*>(x), incx,
                                   reinterpret_cast<cuDoubleComplex*>(y), incy),
                       "cublasZaxpy");
        else
          cublas_check(cublasCaxpy(bref.handle, n, reinterpret_cast<const cuComplex*>(&a),
                                   reinterpret_cast<const cuComplex*>(x), incx,
                                   reinterpret_cast<cuComplex*>(y), incy),
                       "cublasCaxpy");
      }

      static void raxpy(BRef bref, int n, Real a, const Scalar* x, int incx, Scalar* y, int incy) {
        if constexpr (!is_complex) {
          axpy(bref, n, a, x, incx, y, incy);
        } else {
          axpy(bref, n, Scalar(a, 0), x, incx, y, incy);
        }
      }

      static Real nrm2(BRef bref, int n, const Scalar* x, int incx) {
        Real result{};
        if constexpr (std::is_same_v<Scalar, double>)
          cublas_check(cublasDnrm2(bref.handle, n, x, incx, &result), "cublasDnrm2");
        else if constexpr (std::is_same_v<Scalar, float>)
          cublas_check(cublasSnrm2(bref.handle, n, x, incx, &result), "cublasSnrm2");
        else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
          cublas_check(cublasDznrm2(bref.handle, n, reinterpret_cast<const cuDoubleComplex*>(x), incx, &result), "cublasDznrm2");
        else
          cublas_check(cublasScnrm2(bref.handle, n, reinterpret_cast<const cuComplex*>(x), incx, &result), "cublasScnrm2");
        return result;
      }

      static Scalar dot(BRef bref, int n, const Scalar* x, int incx, const Scalar* y, int incy) {
        // Only meaningful for real Scalar; the sym/herm path uses dotc/rdotc.
        Scalar result{};
        if constexpr (std::is_same_v<Scalar, double>)
          cublas_check(cublasDdot(bref.handle, n, x, incx, y, incy, &result), "cublasDdot");
        else if constexpr (std::is_same_v<Scalar, float>)
          cublas_check(cublasSdot(bref.handle, n, x, incx, y, incy, &result), "cublasSdot");
        return result;
      }

      static Scalar dotc(BRef bref, int n, const Scalar* x, int incx, const Scalar* y, int incy) {
        Scalar result{};
        if constexpr (std::is_same_v<Scalar, double>) {
          cublas_check(cublasDdot(bref.handle, n, x, incx, y, incy, &result), "cublasDdot");
        } else if constexpr (std::is_same_v<Scalar, float>) {
          cublas_check(cublasSdot(bref.handle, n, x, incx, y, incy, &result), "cublasSdot");
        } else if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
          cublas_check(cublasZdotc(bref.handle, n, reinterpret_cast<const cuDoubleComplex*>(x), incx,
                                   reinterpret_cast<const cuDoubleComplex*>(y), incy, reinterpret_cast<cuDoubleComplex*>(&result)),
                       "cublasZdotc");
        } else {
          cublas_check(cublasCdotc(bref.handle, n, reinterpret_cast<const cuComplex*>(x), incx,
                                   reinterpret_cast<const cuComplex*>(y), incy, reinterpret_cast<cuComplex*>(&result)),
                       "cublasCdotc");
        }
        return result;
      }

      static Real rdotc(BRef bref, int n, const Scalar* x, int incx, const Scalar* y, int incy) {
        return std::real(dotc(bref, n, x, incx, y, incy));
      }

      static const char* herm_trans() {
        if constexpr (is_complex) return "C";
        else                      return "T";
      }

      // ---- BLAS-2 / BLAS-3 --------------------------------------------------
      static void gemv(BRef bref, const char* trans, int m, int n, Scalar alpha, const Scalar* a, int lda, const Scalar* x,
                       int incx, Scalar beta, Scalar* y, int incy) {
        auto op = cublas_op(trans);
        if constexpr (std::is_same_v<Scalar, double>)
          cublas_check(cublasDgemv(bref.handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy), "cublasDgemv");
        else if constexpr (std::is_same_v<Scalar, float>)
          cublas_check(cublasSgemv(bref.handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy), "cublasSgemv");
        else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
          cublas_check(cublasZgemv(bref.handle, op, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                   reinterpret_cast<const cuDoubleComplex*>(a), lda,
                                   reinterpret_cast<const cuDoubleComplex*>(x), incx,
                                   reinterpret_cast<const cuDoubleComplex*>(&beta),
                                   reinterpret_cast<cuDoubleComplex*>(y), incy),
                       "cublasZgemv");
        else
          cublas_check(cublasCgemv(bref.handle, op, m, n, reinterpret_cast<const cuComplex*>(&alpha),
                                   reinterpret_cast<const cuComplex*>(a), lda, reinterpret_cast<const cuComplex*>(x), incx,
                                   reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<cuComplex*>(y), incy),
                       "cublasCgemv");
      }

      static void gemm(BRef bref, const char* transa, const char* transb, int m, int n, int k, Scalar alpha, const Scalar* a,
                       int lda, const Scalar* b, int ldb, Scalar beta, Scalar* c, int ldc) {
        auto opa = cublas_op(transa);
        auto opb = cublas_op(transb);
        if constexpr (std::is_same_v<Scalar, double>)
          cublas_check(cublasDgemm(bref.handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc), "cublasDgemm");
        else if constexpr (std::is_same_v<Scalar, float>)
          cublas_check(cublasSgemm(bref.handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc), "cublasSgemm");
        else if constexpr (std::is_same_v<Scalar, std::complex<double>>)
          cublas_check(cublasZgemm(bref.handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                   reinterpret_cast<const cuDoubleComplex*>(a), lda,
                                   reinterpret_cast<const cuDoubleComplex*>(b), ldb,
                                   reinterpret_cast<const cuDoubleComplex*>(&beta),
                                   reinterpret_cast<cuDoubleComplex*>(c), ldc),
                       "cublasZgemm");
        else
          cublas_check(cublasCgemm(bref.handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha),
                                   reinterpret_cast<const cuComplex*>(a), lda, reinterpret_cast<const cuComplex*>(b), ldb,
                                   reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<cuComplex*>(c), ldc),
                       "cublasCgemm");
      }

      // gemv with a real x-vector applied to a possibly-complex matrix.
      // The Real x in callers (sapps, seupd) always lives in host workl/h/q.
      // Under CudaBackend we gather x into a contiguous Scalar host buffer
      // (widening on the complex path), upload to a device temp, then call
      // the regular gemv. y is device-resident (workd / V column).
      static void gemv_rv(BRef bref, const char* trans, int m, int ncols, Real alpha, const Scalar* a, int lda, const Real* x,
                          int incx, Real beta, Scalar* y, int incy) {
        int                 xlen = (*trans == 'N' || *trans == 'n') ? ncols : m;
        std::vector<Scalar> host_x(xlen);
        if constexpr (!is_complex) {
          for (int i = 0; i < xlen; ++i) host_x[i] = x[i * incx];
        } else {
          for (int i = 0; i < xlen; ++i) host_x[i] = Scalar(x[i * incx], 0);
        }
        Scalar* dev_x = nullptr;
        cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&dev_x), static_cast<std::size_t>(xlen) * sizeof(Scalar), bref.stream),
                   "cudaMallocAsync(gemv_rv temp)");
        cuda_check(cudaMemcpyAsync(dev_x, host_x.data(), static_cast<std::size_t>(xlen) * sizeof(Scalar), cudaMemcpyHostToDevice,
                                   bref.stream),
                   "cudaMemcpyAsync(gemv_rv H2D)");
        // Sync so host_x outlives the upload (host_x is local).
        cuda_check(cudaStreamSynchronize(bref.stream), "cudaStreamSynchronize(gemv_rv upload)");
        if constexpr (!is_complex) {
          gemv(bref, trans, m, ncols, alpha, a, lda, dev_x, 1, beta, y, incy);
        } else {
          gemv(bref, trans, m, ncols, Scalar(alpha, 0), a, lda, dev_x, 1, Scalar(beta, 0), y, incy);
        }
        cuda_check(cudaFreeAsync(dev_x, bref.stream), "cudaFreeAsync(gemv_rv temp)");
      }

      // ---- LAPACK aux that operate on length-n device data ------------------

      // larnv: generate random vector on host via the CPU Ops, then upload.
      // Avoids the bookkeeping of cuRAND for a once-per-restart-attempt call.
      static void larnv(BRef bref, int idist, int* iseed, int n, Scalar* x) {
        std::vector<Scalar> host_x(n);
        Ops<Scalar, CpuBackend>::larnv(idist, iseed, n, host_x.data());
        cuda_check(cudaMemcpyAsync(x, host_x.data(), static_cast<std::size_t>(n) * sizeof(Scalar), cudaMemcpyHostToDevice,
                                   bref.stream),
                   "cudaMemcpyAsync(larnv H2D)");
        cuda_check(cudaStreamSynchronize(bref.stream), "cudaStreamSynchronize(larnv)");
      }

      // lascl_: scale a device-resident block by cto/cfrom, applying the
      // overflow-safe multi-step ratio that LAPACK dlascl uses (this matters
      // because the only caller triggers it when cfrom = rnorm < safmin, so
      // 1/cfrom would overflow). The mul sequence is pure scalar arithmetic
      // computed on the host; each step is applied on-device with rscal
      // (a real scaling — cublasXscal / Zdscal / Csscal), keeping the path
      // nvcc-free. Only type 'G' (general, full m x n block) is supported,
      // which is the only type any caller reaching this backend uses; other
      // types fall back to the host path.
      static void lascl_(BRef bref, const char* type, const int* kl, const int* ku, const Real* cfrom, const Real* cto,
                         const int* m, const int* n, Scalar* a, const int* lda, int* info) {
        *info = 0;
        if (!(type[0] == 'G' || type[0] == 'g')) {
          // Unexpected type (none of the algorithmic call sites use these
          // under a device backend) — preserve correctness via host fallback.
          const int           nelem = (*lda) * (*n);
          std::vector<Scalar> host_a(nelem);
          cuda_check(cudaMemcpyAsync(host_a.data(), a, static_cast<std::size_t>(nelem) * sizeof(Scalar), cudaMemcpyDeviceToHost,
                                     bref.stream),
                     "cudaMemcpyAsync(lascl D2H)");
          cuda_check(cudaStreamSynchronize(bref.stream), "cudaStreamSynchronize(lascl D2H)");
          Ops<Scalar, CpuBackend>::lascl_(type, kl, ku, cfrom, cto, m, n, host_a.data(), lda, info);
          cuda_check(cudaMemcpyAsync(a, host_a.data(), static_cast<std::size_t>(nelem) * sizeof(Scalar), cudaMemcpyHostToDevice,
                                     bref.stream),
                     "cudaMemcpyAsync(lascl H2D)");
          cuda_check(cudaStreamSynchronize(bref.stream), "cudaStreamSynchronize(lascl H2D)");
          return;
        }

        const Real smlnum = Ops<Real, CpuBackend>::lamch("safe minimum");
        const Real bignum = Real(1) / smlnum;
        Real       cfromc = *cfrom;
        Real       ctoc   = *cto;
        const int  rows   = *m;
        const int  cols   = *n;
        const int  ld     = *lda;

        bool done = false;
        while (!done) {
          Real cfrom1 = cfromc * smlnum;
          Real mul;
          if (cfrom1 == cfromc) {
            // cfromc is an infinity; cto/cfrom is the well-defined ratio.
            mul  = ctoc / cfromc;
            done = true;
          } else {
            Real cto1 = ctoc / bignum;
            if (cto1 == ctoc) {
              // ctoc is zero.
              mul    = ctoc;
              done   = true;
              cfromc = Real(1);
            } else if (std::abs(cfrom1) > std::abs(ctoc) && ctoc != Real(0)) {
              mul    = smlnum;
              done   = false;
              cfromc = cfrom1;
            } else if (std::abs(cto1) > std::abs(cfromc)) {
              mul    = bignum;
              done   = false;
              ctoc   = cto1;
            } else {
              mul  = ctoc / cfromc;
              done = true;
            }
          }
          // Apply A *= mul over the general m x n block (column-major, ld).
          for (int j = 0; j < cols; ++j)
            rscal(bref, rows, mul, a + static_cast<std::size_t>(j) * ld, 1);
        }
      }
    };

  }  // namespace detail

}  // namespace arnoldi

#endif  // ARNOLDI_USE_CUDA

#endif  // ARNOLDI_CUDA_HPP
