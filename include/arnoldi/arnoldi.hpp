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

  template <Kind K, typename Scalar, typename Comm = SerialComm>
  class Arnoldi {
  public:
    using real_type   = typename detail::real_of<Scalar>::type;
    using scalar_type = Scalar;

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

  public:
    Arnoldi(std::string bmat, int n, std::string which, int nev, int ncv, Comm comm = Comm{}) :
        bmat_(std::move(bmat)), which_(std::move(which)), n_(n), nev_(nev), ncv_(ncv), ldv_(n), comm_(std::move(comm)) {
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

    Arnoldi& initial_resid(const Scalar* r) {
      std::copy(r, r + n_, resid_.begin());
      info_in_ = 1;
      return *this;
    }

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

    bool             converged() const noexcept { return info_ == 0 && iparam_[4] >= nev_; }
    int              num_converged() const noexcept { return iparam_[4]; }
    int              num_iterations() const noexcept { return iparam_[2]; }
    int              num_op_applies() const noexcept { return iparam_[8]; }
    int              info() const noexcept { return info_; }

    const Scalar*    v() const noexcept { return v_.data(); }
    const real_type* workl() const noexcept { return workl_.data(); }
    const int*       iparam() const noexcept { return iparam_.data(); }
    const int*       ipntr() const noexcept { return ipntr_.data(); }

  private:
    template <class Op, class Bop>
    void dispatch_aupd_(Op&& op, Bop&& bop) {
      if constexpr (K == Kind::Nonsym) {
        arnoldi::detail::naupd<real_type>(bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(), ldv_,
                                      iparam_.data(), ipntr_.data(), workd_.data(), workl_.data(), lworkl_, info_,
                                      std::forward<Op>(op), std::forward<Bop>(bop), comm_);
      } else {  // Sym or Herm
        arnoldi::detail::saupd<Scalar>(bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(), ldv_,
                                   iparam_.data(), ipntr_.data(), workd_.data(), workl_.data(), lworkl_, info_,
                                   std::forward<Op>(op), std::forward<Bop>(bop), comm_);
      }
    }

    template <class Op>
    void dispatch_aupd_(Op&& op) {
      if constexpr (K == Kind::Nonsym) {
        arnoldi::detail::naupd<real_type>(
            bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(), ldv_, iparam_.data(), ipntr_.data(),
            workd_.data(), workl_.data(), lworkl_, info_, std::forward<Op>(op), [](const real_type*, real_type*) {}, comm_);
      } else {
        arnoldi::detail::saupd<Scalar>(
            bmat_.c_str(), n_, which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(), ldv_, iparam_.data(), ipntr_.data(),
            workd_.data(), workl_.data(), lworkl_, info_, std::forward<Op>(op), [](const Scalar*, Scalar*) {}, comm_);
      }
    }

    Result extract_(bool rvec, real_type sigma) {
      Result out{};
      out.values.assign(static_cast<std::size_t>(nev_), real_type{});
      if (rvec) out.vectors.assign(static_cast<std::size_t>(n_) * nev_, Scalar{});

      arnoldi::detail::seupd<Scalar>(rvec, "A", out.values.data(), rvec ? out.vectors.data() : nullptr, n_, sigma, bmat_.c_str(), n_,
                                 which_.c_str(), nev_, tol_, resid_.data(), ncv_, v_.data(), ldv_, iparam_.data(), ipntr_.data(),
                                 workd_.data(), workl_.data(), lworkl_, info_, comm_);
      return out;
    }

    Result extract_(bool rvec, real_type sigmar, real_type sigmai) {
      Result out{};
      // dnaupd returns up to nev+1 eigenvalue slots (for the possible extra
      // conjugate pair partner); eigenvector matrix matches width.
      out.values_re.assign(static_cast<std::size_t>(nev_ + 1), real_type{});
      out.values_im.assign(static_cast<std::size_t>(nev_ + 1), real_type{});
      if (rvec) out.vectors.assign(static_cast<std::size_t>(n_) * (nev_ + 1), Scalar{});

      arnoldi::detail::neupd<real_type>(rvec, "A", out.values_re.data(), out.values_im.data(), rvec ? out.vectors.data() : nullptr,
                                    n_, sigmar, sigmai, workev_.data(), bmat_.c_str(), n_, which_.c_str(), nev_, tol_,
                                    resid_.data(), ncv_, v_.data(), ldv_, iparam_.data(), ipntr_.data(), workd_.data(),
                                    workl_.data(), lworkl_, info_);
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

    std::vector<scalar_type> resid_;
    std::vector<scalar_type> v_;
    std::vector<scalar_type> workd_;
    std::vector<real_type>   workl_;
    std::vector<real_type>   workev_;  // Nonsym only

    std::array<int, 11>      iparam_{};
    std::array<int, 14>      ipntr_{};
  };

  template <class R, class C = SerialComm>
  using SymArnoldi = Arnoldi<Kind::Sym, R, C>;
  template <class R, class C = SerialComm>
  using NonsymArnoldi = Arnoldi<Kind::Nonsym, R, C>;
  template <class R, class C = SerialComm>
  using HermArnoldi = Arnoldi<Kind::Herm, std::complex<R>, C>;

}  // namespace arnoldi

#endif  // ARNOLDI_ARNOLDI_HPP
