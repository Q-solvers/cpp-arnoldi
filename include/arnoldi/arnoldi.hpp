#ifndef ARNOLDI_ARNOLDI_HPP
#define ARNOLDI_ARNOLDI_HPP

// Arnoldi<K, Scalar> — callback-style ARPACK driver class.
//
//   Kind::Sym    + real Scalar          → dsaupd/dseupd, ssaupd/sseupd
//   Kind::Nonsym + real Scalar          → dnaupd/dneupd, snaupd/sneupd
//                                         (complex eigenvalues returned as
//                                          conjugate pairs in two real arrays)
//   Kind::Herm   + std::complex<Real>   → Hermitian via saupd<complex<Real>>
//
// Usage:
//   arnoldi::Arnoldi<arnoldi::Kind::Sym, double> s("I", n, "LM", nev, ncv);
//   s.tol(1e-10).maxiter(500);
//   s.solve([&](const double* x, double* y) { /* y = A*x */ });
//   auto r = s.eigenpairs();          // r.values, r.vectors
//
// See examples/ for one demo per Kind.

#include <arnoldi/comm.hpp>
#include <arnoldi/detail.hpp>
#include <array>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace arnoldi {

  enum class Kind { Sym, Nonsym, Herm };

  namespace detail {

    template <class T>
    struct is_complex : std::false_type {};
    template <class T>
    struct is_complex<std::complex<T>> : std::true_type {};

    template <class T>
    struct real_of {
      using type = T;
    };
    template <class T>
    struct real_of<std::complex<T>> {
      using type = T;
    };

    template <class T>
    inline constexpr bool is_complex_v = is_complex<T>::value;

  }  // namespace detail

  template <Kind K, typename Scalar, typename Comm = SerialComm, typename Backend = arnoldi::detail::CpuBackend>
  class Arnoldi {
  public:
    using real_type   = typename detail::real_of<Scalar>::type;
    using scalar_type = Scalar;
    using backend_type = Backend;

  private:
    static_assert(std::is_floating_point_v<real_type>,
                  "Scalar must be float, double, std::complex<float>, or std::complex<double>");

    static_assert(K != Kind::Sym || !detail::is_complex_v<Scalar>,
                  "Kind::Sym requires a real Scalar (use Kind::Herm for complex Hermitian)");

    static_assert(K != Kind::Nonsym || !detail::is_complex_v<Scalar>,
                  "Kind::Nonsym requires a real Scalar "
                  "(complex nonsymmetric znaupd/cnaupd is not implemented in arpack_callback)");

    static_assert(K != Kind::Herm || detail::is_complex_v<Scalar>,
                  "Kind::Herm requires a complex Scalar (std::complex<float/double>)");

    // The nonsymmetric path goes through real-Schur LAPACK auxiliaries
    // (dlahqr_/dtrsen_/dtrevc_) that have no cuSOLVER analogue for
    // Hessenberg matrices. Only Sym / Herm are supported under device
    // backends; reject Nonsym + non-CpuBackend at compile time.
    static_assert(K != Kind::Nonsym || std::is_same_v<Backend, arnoldi::detail::CpuBackend>,
                  "Kind::Nonsym requires CpuBackend (no device specialization for the real-Schur LAPACK aux)");

  public:
    Arnoldi(std::string bmat, int n, std::string which, int nev, int ncv, Comm comm = Comm{}, Backend backend = Backend{}) :
        bmat_(std::move(bmat)), which_(std::move(which)), n_(n), nev_(nev), ncv_(ncv), ldv_(n), comm_(std::move(comm)),
        backend_(std::move(backend)) {
      if (n_ <= 0) throw std::invalid_argument("Arnoldi: n must be > 0");
      if (nev_ <= 0) throw std::invalid_argument("Arnoldi: nev must be > 0");
      int n_global = comm_.allreduce_sum(n_);
      if (ncv_ > n_global) throw std::invalid_argument("Arnoldi: ncv must be <= global dimension N");
      if (bmat_ != "I" && bmat_ != "G") throw std::invalid_argument("Arnoldi: bmat must be \"I\" or \"G\"");
      if constexpr (K == Kind::Nonsym) {
        if (which_ != "LM" && which_ != "SM" && which_ != "LR" && which_ != "SR" && which_ != "LI" && which_ != "SI")
          throw std::invalid_argument("Arnoldi<Nonsym>: which must be one of LM, SM, LR, SR, LI, SI");
      } else {
        if (which_ != "LM" && which_ != "SM" && which_ != "LA" && which_ != "SA" && which_ != "BE")
          throw std::invalid_argument("Arnoldi<Sym/Herm>: which must be one of LM, SM, LA, SA, BE");
      }

      if constexpr (K == Kind::Nonsym) {
        if (ncv_ <= nev_ + 1) throw std::invalid_argument("Arnoldi<Nonsym>: ncv must satisfy ncv > nev+1");
      } else {
        if (ncv_ <= nev_) throw std::invalid_argument("Arnoldi<Sym/Herm>: ncv must satisfy ncv > nev");
      }

      resid_.assign(static_cast<std::size_t>(n_), Scalar{});
      v_.assign(static_cast<std::size_t>(ldv_) * ncv_, Scalar{});
      workd_.assign(static_cast<std::size_t>(3 * n_), Scalar{});

      if constexpr (K == Kind::Nonsym) {
        lworkl_ = 3 * ncv_ * ncv_ + 6 * ncv_;
        workev_.assign(static_cast<std::size_t>(3 * ncv_), real_type{});
      } else {
        lworkl_ = ncv_ * ncv_ + 8 * ncv_;
      }
      workl_.assign(static_cast<std::size_t>(lworkl_), real_type{});

      iparam_[0] = 1;
      iparam_[2] = 300;
      iparam_[6] = 1;
    }

    Arnoldi& tol(real_type t) noexcept {
      tol_ = t;
      return *this;
    }
    Arnoldi& maxiter(int m) noexcept {
      iparam_[2] = m;
      return *this;
    }
    Arnoldi& mode(int m) noexcept {
      iparam_[6] = m;
      return *this;
    }
    Arnoldi& ishift(int s) noexcept {
      iparam_[0] = s;
      return *this;
    }

    // Seed the iteration with a user-provided initial residual vector. `r`
    // is a HOST pointer; under CudaBackend the data is staged H->D once,
    // here, into the device-resident resid_ buffer.
    Arnoldi& initial_resid(const Scalar* r) {
      arnoldi::detail::buffer_traits<Backend, scalar_type>::copy_from_host(resid_, r, static_cast<std::size_t>(n_));
      info_in_ = 1;
      return *this;
    }

    // Device-pointer overload: only available under a non-CpuBackend (e.g.
    // CudaBackend). Copies device-to-device into resid_. The device-specific
    // buffer_traits specialization supplies copy_from_device; this member
    // is SFINAE'd out under CpuBackend.
    template <typename B = Backend, std::enable_if_t<!std::is_same_v<B, arnoldi::detail::CpuBackend>, int> = 0>
    Arnoldi& initial_resid_device(const Scalar* r) {
      arnoldi::detail::buffer_traits<B, scalar_type>::copy_from_device(resid_, r, static_cast<std::size_t>(n_));
      info_in_ = 1;
      return *this;
    }

    // Accessor returning a reference to the owned backend. Users invoking
    // a GPU matvec from inside their callback can grab the cuBLAS handle
    // and stream via `solver.backend().handle()` / `solver.backend().stream()`.
    Backend&       backend() noexcept { return backend_; }
    const Backend& backend() const noexcept { return backend_; }

    template <class Op>
    void solve(Op&& op) {
      info_ = info_in_;
      dispatch_aupd_(std::forward<Op>(op));
    }

    template <class Op, class Bop>
    void solve(Op&& op, Bop&& bop) {
      info_ = info_in_;
      dispatch_aupd_(std::forward<Op>(op), std::forward<Bop>(bop));
    }

    struct SymResult {
      std::vector<real_type>   values;   // size nev
      std::vector<scalar_type> vectors;  // n × nev (column-major), empty if !rvec
    };
    struct NonsymResult {
      // Eigenvalues come as (re, im); conjugate pairs occupy two adjacent slots.
      std::vector<real_type>   values_re;  // size nev+1
      std::vector<real_type>   values_im;  // size nev+1
      // Eigenvectors: real Schur basis in ldv × (nev+1).  For a conjugate
      // pair (values_im[i] != 0), columns i and i+1 hold Re and Im parts.
      std::vector<scalar_type> vectors;
    };
    struct HermResult {
      std::vector<real_type>   values;   // size nev (Hermitian → real)
      std::vector<scalar_type> vectors;  // n × nev complex
    };

    using Result = std::conditional_t<K == Kind::Sym, SymResult, std::conditional_t<K == Kind::Nonsym, NonsymResult, HermResult>>;

    // Sym / Herm result with eigenvectors left on the device. Eigenvalues are
    // always host-resident (they come out of the ncv-sized Hessenberg
    // subproblem, computed on host). The eigenvector matrix is the device
    // buffer seupd's final GEMM wrote into — handed back without the n×nev
    // device→host copy that eigenpairs() performs.
    struct DeviceResult {
      std::vector<real_type>                                                    values;   // size nev (host)
      typename arnoldi::detail::buffer_traits<Backend, scalar_type>::vector_type vectors;  // n × nev (device)
    };

    // Real shift (Sym / Herm only).
    template <Kind KK = K, std::enable_if_t<KK != Kind::Nonsym, int> = 0>
    Result eigenpairs(bool compute_vectors = true, real_type sigma = real_type{}) {
      return extract_(compute_vectors, sigma);
    }

    // Nonsym overload: separate real/imag parts of the shift.
    template <Kind KK = K, std::enable_if_t<KK == Kind::Nonsym, int> = 0>
    Result eigenpairs(bool compute_vectors = true, real_type sigmar = real_type{}, real_type sigmai = real_type{}) {
      return extract_(compute_vectors, sigmar, sigmai);
    }

    // Sym / Herm only, device backends only: extract eigenpairs leaving the
    // eigenvector matrix on the device (column-major, n × nev). Use this when
    // downstream work consumes the eigenvectors on the GPU and the host copy
    // that eigenpairs() does would be wasted.
    template <Kind KK = K, typename B = Backend,
              std::enable_if_t<KK != Kind::Nonsym && !std::is_same_v<B, arnoldi::detail::CpuBackend>, int> = 0>
    DeviceResult eigenpairs_device(bool compute_vectors = true, real_type sigma = real_type{}) {
      DeviceResult out{};
      out.values.assign(static_cast<std::size_t>(nev_), real_type{});
      if (compute_vectors) out.vectors.assign(static_cast<std::size_t>(n_) * nev_, Scalar{});

      auto bref = backend_.ref();
      arnoldi::detail::seupd<Scalar, Backend>(bref, compute_vectors, "A", out.values.data(),
                                              compute_vectors ? out.vectors.data() : nullptr, n_, sigma, bmat_.c_str(), n_,
                                              which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(), ldv_, iparam_.data(),
                                              ipntr_.data(), workd_.data(), workl_.data(), lworkl_, info_, comm_);
      return out;
    }

    bool             converged() const noexcept { return info_ == 0 && iparam_[4] >= nev_; }
    int              num_converged() const noexcept { return iparam_[4]; }
    int              num_iterations() const noexcept { return iparam_[2]; }
    int              num_op_applies() const noexcept { return iparam_[8]; }
    int              info() const noexcept { return info_; }

    const Scalar*    v() const noexcept { return v_.data(); }
    Scalar*          workd() noexcept { return workd_.data(); }
    const real_type* workl() const noexcept { return workl_.data(); }
    const int*       iparam() const noexcept { return iparam_.data(); }
    const int*       ipntr() const noexcept { return ipntr_.data(); }

  private:
    template <class Op, class Bop>
    void dispatch_aupd_(Op&& op, Bop&& bop) {
      auto bref = backend_.ref();
      if constexpr (K == Kind::Nonsym) {
        arnoldi::detail::naupd<real_type, Backend>(bref, bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_,
                                                   v_.data(), ldv_, iparam_.data(), ipntr_.data(), workd_.data(), workl_.data(),
                                                   lworkl_, info_, std::forward<Op>(op), std::forward<Bop>(bop), comm_);
      } else {  // Sym or Herm
        arnoldi::detail::saupd<Scalar, Backend>(bref, bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_,
                                                v_.data(), ldv_, iparam_.data(), ipntr_.data(), workd_.data(), workl_.data(),
                                                lworkl_, info_, std::forward<Op>(op), std::forward<Bop>(bop), comm_);
      }
    }

    template <class Op>
    void dispatch_aupd_(Op&& op) {
      auto bref = backend_.ref();
      if constexpr (K == Kind::Nonsym) {
        arnoldi::detail::naupd<real_type, Backend>(
            bref, bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(), ldv_, iparam_.data(),
            ipntr_.data(), workd_.data(), workl_.data(), lworkl_, info_, std::forward<Op>(op),
            [](const real_type*, real_type*) {}, comm_);
      } else {
        arnoldi::detail::saupd<Scalar, Backend>(
            bref, bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(), ldv_, iparam_.data(),
            ipntr_.data(), workd_.data(), workl_.data(), lworkl_, info_, std::forward<Op>(op), [](const Scalar*, Scalar*) {},
            comm_);
      }
    }

    Result extract_(bool rvec, real_type sigma) {
      Result out{};
      out.values.assign(static_cast<std::size_t>(nev_), real_type{});
      if (rvec) out.vectors.assign(static_cast<std::size_t>(n_) * nev_, Scalar{});

      auto bref = backend_.ref();
      if constexpr (std::is_same_v<Backend, arnoldi::detail::CpuBackend>) {
        // CPU: seupd writes the final V*S directly into host out.vectors.
        arnoldi::detail::seupd<Scalar, Backend>(bref, rvec, "A", out.values.data(), rvec ? out.vectors.data() : nullptr, n_, sigma,
                                                bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(),
                                                ldv_, iparam_.data(), ipntr_.data(), workd_.data(), workl_.data(), lworkl_, info_,
                                                comm_);
      } else {
        // Device: allocate a device buffer for z, run seupd into it (final
        // GEMM lands on device), then copy down to host out.vectors. The
        // matching `buffer_copy_to_host` and `buffer_traits` specialisations
        // come from the device-specific backend header (e.g. cuda.hpp).
        typename arnoldi::detail::buffer_traits<Backend, scalar_type>::vector_type z_dev;
        if (rvec) z_dev.assign(static_cast<std::size_t>(n_) * nev_, Scalar{});
        arnoldi::detail::seupd<Scalar, Backend>(bref, rvec, "A", out.values.data(), rvec ? z_dev.data() : nullptr, n_, sigma,
                                                bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(),
                                                ldv_, iparam_.data(), ipntr_.data(), workd_.data(), workl_.data(), lworkl_, info_,
                                                comm_);
        if (rvec)
          arnoldi::detail::buffer_traits<Backend, scalar_type>::copy_to_host(out.vectors.data(), z_dev,
                                                                              static_cast<std::size_t>(n_) * nev_);
      }
      return out;
    }

    Result extract_(bool rvec, real_type sigmar, real_type sigmai) {
      Result out{};
      // dnaupd returns up to nev+1 eigenvalue slots (for the possible extra
      // conjugate pair partner); eigenvector matrix matches width.
      out.values_re.assign(static_cast<std::size_t>(nev_ + 1), real_type{});
      out.values_im.assign(static_cast<std::size_t>(nev_ + 1), real_type{});
      if (rvec) out.vectors.assign(static_cast<std::size_t>(n_) * (nev_ + 1), Scalar{});

      auto bref = backend_.ref();
      arnoldi::detail::neupd<real_type, Backend>(bref, rvec, "A", out.values_re.data(), out.values_im.data(),
                                                 rvec ? out.vectors.data() : nullptr, n_, sigmar, sigmai, workev_.data(),
                                                 bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(),
                                                 ldv_, iparam_.data(), ipntr_.data(), workd_.data(), workl_.data(), lworkl_, info_);
      return out;
    }

    std::string              bmat_;
    std::string              which_;
    int                      n_, nev_, ncv_, ldv_;
    int                      lworkl_  = 0;
    int                      info_    = 0;
    int                      info_in_ = 0;  // initial value fed into *aupd (1 if user gave resid)
    real_type                tol_     = real_type{};
    Comm                     comm_;
    Backend                  backend_;

    // Length-n workspace lives on the backend (host std::vector under
    // CpuBackend, device_vector under CudaBackend).
    typename arnoldi::detail::buffer_traits<Backend, scalar_type>::vector_type resid_;
    typename arnoldi::detail::buffer_traits<Backend, scalar_type>::vector_type v_;
    typename arnoldi::detail::buffer_traits<Backend, scalar_type>::vector_type workd_;
    // The IRA bookkeeping (Hessenberg, Ritz values/bounds, Givens scratch)
    // is small (ncv²-sized) and stays host-resident under every backend.
    std::vector<real_type>   workl_;
    std::vector<real_type>   workev_;  // Nonsym only

    std::array<int, 11>      iparam_{};
    std::array<int, 14>      ipntr_{};
  };

  template <class R, class C = SerialComm, class B = arnoldi::detail::CpuBackend>
  using SymArnoldi = Arnoldi<Kind::Sym, R, C, B>;
  template <class R, class C = SerialComm, class B = arnoldi::detail::CpuBackend>
  using NonsymArnoldi = Arnoldi<Kind::Nonsym, R, C, B>;
  template <class R, class C = SerialComm, class B = arnoldi::detail::CpuBackend>
  using HermArnoldi = Arnoldi<Kind::Herm, std::complex<R>, C, B>;

}  // namespace arnoldi

#endif  // ARNOLDI_ARNOLDI_HPP
