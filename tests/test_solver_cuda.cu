// CUDA-backend tests for arnoldi::Arnoldi.
//
// Mirror of tests/test_solver.cpp for the Sym / Herm path, but with the
// matvec implemented as a CUDA kernel receiving device pointers. The user
// matvec callback receives DEVICE pointers (per the cuda.hpp contract);
// solver workspace lives in device_vector under CudaBackend.
//
// Cases:
//   - Sym Laplacian (n=128): eigenvalues match the exact analytical formula
//     2 - 2 cos(k pi / (n+1)).
//   - Herm complex tridiagonal (n=128): eigenvalues real, residual
//     ||A v - lambda v|| / ||A|| below tolerance.
//   - which="LM": largest-magnitude eigenvalue is large.
//   - Eigenvector residual: ||A v - lambda v|| is small.
//   - initial_resid_device: solve seeded from a device-resident vector.

#include <arnoldi/arnoldi.hpp>
#include <arnoldi/cuda.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <vector>

#include <catch2/catch_test_macros.hpp>

// Bridge legacy check("desc", cond) calls onto Catch2 assertions.
#define check(msg, cond) \
    do {                 \
        INFO(msg);       \
        CHECK((cond));   \
    } while (0)

// Skip a CUDA test case gracefully when no device is present.
static bool cuda_device_present() {
    int c = 0;
    return cudaGetDeviceCount(&c) == cudaSuccess && c > 0;
}
#define REQUIRE_CUDA()                                  \
    do {                                                \
        if (!cuda_device_present())                     \
            SKIP("no CUDA device available");           \
    } while (0)

static double exact_eig_laplacian(int n, int k) {
    return 2.0 - 2.0 * std::cos(k * M_PI / (n + 1));
}

// -------- Kernels ------------------------------------------------------------
// 1D Dirichlet Laplacian:  y = A x  with A(i,i)=2, A(i,i±1)=-1.
__global__ void d_av_laplacian(int n, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double left  = (i == 0)     ? 0.0 : x[i - 1];
    double right = (i == n - 1) ? 0.0 : x[i + 1];
    y[i] = 2.0 * x[i] - left - right;
}

// Hermitian complex tridiagonal:
//   H(i,i)   = 2
//   H(i,i+1) = off          (and H(i+1,i) = conj(off))
// with off = -1 + 0.25 i. The eigenvalues are real because H is Hermitian.
__global__ void d_av_hermitian(int n, const cuDoubleComplex* x, cuDoubleComplex* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const cuDoubleComplex off      = make_cuDoubleComplex(-1.0, 0.25);
    const cuDoubleComplex off_conj = make_cuDoubleComplex(-1.0, -0.25);
    cuDoubleComplex left  = (i == 0)     ? make_cuDoubleComplex(0.0, 0.0) : cuCmul(off_conj, x[i - 1]);
    cuDoubleComplex right = (i == n - 1) ? make_cuDoubleComplex(0.0, 0.0) : cuCmul(off,      x[i + 1]);
    cuDoubleComplex diag  = make_cuDoubleComplex(2.0 * cuCreal(x[i]), 2.0 * cuCimag(x[i]));
    y[i] = cuCadd(cuCadd(left, diag), right);
}

static int blocks_for(int n, int threads = 256) { return (n + threads - 1) / threads; }

// -------- Tests --------------------------------------------------------------

TEST_CASE("test_sym_cuda_laplacian", "[cuda]") {
    REQUIRE_CUDA();
    const int n = 128, nev = 4, ncv = 12;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double, arnoldi::SerialComm,
                     arnoldi::CudaBackend> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    cudaStream_t stream = s.backend().stream();

    s.solve([&](const double* x, double* y) {
        d_av_laplacian<<<blocks_for(n), 256, 0, stream>>>(n, x, y);
    });

    check("Sym<CUDA>: converged", s.converged());
    check("Sym<CUDA>: nconv >= nev", s.num_converged() >= nev);
    check("Sym<CUDA>: info == 0", s.info() == 0);

    auto r = s.eigenpairs(false);
    // seupd does not guarantee a particular order within the converged set;
    // compare as sorted sets. The nev smallest exact Laplacian eigenvalues
    // are exact_eig_laplacian(n, 1..nev).
    std::vector<double> got(r.values.begin(), r.values.end());
    std::sort(got.begin(), got.end());
    std::vector<double> want(nev);
    for (int k = 0; k < nev; ++k) want[k] = exact_eig_laplacian(n, k + 1);
    std::sort(want.begin(), want.end());
    double max_err = 0.0;
    for (int k = 0; k < nev; ++k) max_err = std::max(max_err, std::abs(got[k] - want[k]));
    check("Sym<CUDA>: eigenvalues match exact to 1e-10", max_err < 1e-10);
}

TEST_CASE("test_sym_cuda_which_LM", "[cuda]") {
    REQUIRE_CUDA();
    const int n = 64, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double, arnoldi::SerialComm,
                     arnoldi::CudaBackend> s("I", n, "LM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    cudaStream_t stream = s.backend().stream();

    s.solve([&](const double* x, double* y) {
        d_av_laplacian<<<blocks_for(n), 256, 0, stream>>>(n, x, y);
    });

    check("Sym<CUDA> which=LM: converged", s.converged());
    auto r = s.eigenpairs(false);
    check("Sym<CUDA> which=LM: largest eig > 3.5", r.values[0] > 3.5);
}

TEST_CASE("test_sym_cuda_eigenvectors", "[cuda]") {
    REQUIRE_CUDA();
    const int n = 64, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double, arnoldi::SerialComm,
                     arnoldi::CudaBackend> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    cudaStream_t stream = s.backend().stream();

    s.solve([&](const double* x, double* y) {
        d_av_laplacian<<<blocks_for(n), 256, 0, stream>>>(n, x, y);
    });

    auto r = s.eigenpairs(true);
    check("Sym<CUDA> evecs: vectors.size() == n*nev", (int)r.vectors.size() == n * nev);

    // Residual check: for each (lambda, v), compute ||A v - lambda v|| / ||v||
    // on the host (host-side A applied to the host-resident eigenvectors).
    double max_res = 0.0;
    std::vector<double> Av(n);
    for (int k = 0; k < nev; ++k) {
        const double* vk = &r.vectors[k * n];
        Av[0] = 2.0 * vk[0] - vk[1];
        for (int i = 1; i < n - 1; ++i) Av[i] = -vk[i - 1] + 2.0 * vk[i] - vk[i + 1];
        Av[n - 1] = -vk[n - 2] + 2.0 * vk[n - 1];

        double rnorm = 0.0, vnorm = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = Av[i] - r.values[k] * vk[i];
            rnorm += d * d;
            vnorm += vk[i] * vk[i];
        }
        rnorm = std::sqrt(rnorm / vnorm);
        if (rnorm > max_res) max_res = rnorm;
    }
    check("Sym<CUDA> evecs: max residual < 1e-10", max_res < 1e-10);
}

TEST_CASE("test_herm_cuda", "[cuda]") {
    REQUIRE_CUDA();
    using cplx       = std::complex<double>;
    const int n      = 128, nev = 4, ncv = 12;
    arnoldi::Arnoldi<arnoldi::Kind::Herm, cplx, arnoldi::SerialComm,
                     arnoldi::CudaBackend> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    cudaStream_t stream = s.backend().stream();

    s.solve([&](const cplx* x, cplx* y) {
        d_av_hermitian<<<blocks_for(n), 256, 0, stream>>>(
            n,
            reinterpret_cast<const cuDoubleComplex*>(x),
            reinterpret_cast<cuDoubleComplex*>(y));
    });

    check("Herm<CUDA>: converged", s.converged());
    check("Herm<CUDA>: nconv >= nev", s.num_converged() >= nev);

    auto r = s.eigenpairs(true);
    check("Herm<CUDA>: vectors.size() == n*nev", (int)r.vectors.size() == n * nev);

    // Hermitian eigenvalues must be real-valued (the solver returns them
    // as Real, so this is automatic; just verify finiteness).
    for (int k = 0; k < nev; ++k)
        check("Herm<CUDA>: eigenvalue finite", std::isfinite(r.values[k]));

    // Residual: ||H v - lambda v|| / ||v|| on the host.
    const cplx        off(-1.0, 0.25);
    std::vector<cplx> Hv(n);
    double            max_res = 0.0;
    for (int k = 0; k < nev; ++k) {
        const cplx* vk = &r.vectors[k * n];
        Hv[0]          = 2.0 * vk[0] + off * vk[1];
        for (int i = 1; i < n - 1; ++i)
            Hv[i] = std::conj(off) * vk[i - 1] + 2.0 * vk[i] + off * vk[i + 1];
        Hv[n - 1]   = std::conj(off) * vk[n - 2] + 2.0 * vk[n - 1];

        double rnorm = 0.0, vnorm = 0.0;
        for (int i = 0; i < n; ++i) {
            cplx d = Hv[i] - r.values[k] * vk[i];
            rnorm += std::norm(d);
            vnorm += std::norm(vk[i]);
        }
        rnorm = std::sqrt(rnorm / vnorm);
        if (rnorm > max_res) max_res = rnorm;
    }
    check("Herm<CUDA>: max residual < 1e-9", max_res < 1e-9);
}

TEST_CASE("test_initial_resid_device", "[cuda]") {
    REQUIRE_CUDA();
    const int n = 64, nev = 3, ncv = 10;

    // Allocate a device buffer of size n, seed with 1.0 from the host.
    double* d_resid = nullptr;
    cudaMalloc(&d_resid, n * sizeof(double));
    std::vector<double> h_resid(n, 1.0);
    cudaMemcpy(d_resid, h_resid.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    arnoldi::Arnoldi<arnoldi::Kind::Sym, double, arnoldi::SerialComm,
                     arnoldi::CudaBackend> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    s.initial_resid_device(d_resid);  // device pointer overload
    cudaStream_t stream = s.backend().stream();

    s.solve([&](const double* x, double* y) {
        d_av_laplacian<<<blocks_for(n), 256, 0, stream>>>(n, x, y);
    });

    check("initial_resid_device: converged", s.converged());
    auto r = s.eigenpairs(false);
    // Smallest converged eigenvalue (order within the set is unspecified).
    double got_min = *std::min_element(r.values.begin(), r.values.end());
    double err     = std::abs(got_min - exact_eig_laplacian(n, 1));
    check("initial_resid_device: smallest eigenvalue accurate", err < 1e-10);

    cudaFree(d_resid);
}

TEST_CASE("test_eigenpairs_device", "[cuda]") {
    REQUIRE_CUDA();
    const int n = 64, nev = 3, ncv = 10;
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double, arnoldi::SerialComm,
                     arnoldi::CudaBackend> s("I", n, "SM", nev, ncv);
    s.tol(1e-12).maxiter(500);
    cudaStream_t stream = s.backend().stream();

    s.solve([&](const double* x, double* y) {
        d_av_laplacian<<<blocks_for(n), 256, 0, stream>>>(n, x, y);
    });
    check("eigenpairs_device: converged", s.converged());

    // Eigenvectors stay on the device; eigenvalues come back on the host.
    auto dr = s.eigenpairs_device(true);
    check("eigenpairs_device: values.size() == nev", (int)dr.values.size() == nev);

    // Pull the device eigenvectors down ourselves and residual-check.
    std::vector<double> hv(n * nev);
    cudaMemcpy(hv.data(), dr.vectors.data(), n * nev * sizeof(double), cudaMemcpyDeviceToHost);

    double              max_res = 0.0;
    std::vector<double> Av(n);
    for (int k = 0; k < nev; ++k) {
        const double* vk = &hv[k * n];
        Av[0]            = 2.0 * vk[0] - vk[1];
        for (int i = 1; i < n - 1; ++i) Av[i] = -vk[i - 1] + 2.0 * vk[i] - vk[i + 1];
        Av[n - 1]   = -vk[n - 2] + 2.0 * vk[n - 1];
        double rn = 0.0, vn = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = Av[i] - dr.values[k] * vk[i];
            rn += d * d;
            vn += vk[i] * vk[i];
        }
        max_res = std::max(max_res, std::sqrt(rn / vn));
    }
    check("eigenpairs_device: device-evec residual < 1e-10", max_res < 1e-10);
}
