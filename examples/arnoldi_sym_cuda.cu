// arnoldi_sym_cuda — smallest eigenvalues of the 1D Dirichlet Laplacian
// with the matvec running entirely on the GPU.
//
// Demonstrates the CudaBackend device-matvec pattern:
//   - Arnoldi<Kind::Sym, double, SerialComm, CudaBackend> keeps the Lanczos
//     workspace (resid/v/workd) in device memory.
//   - The user matvec callback receives DEVICE pointers and launches a CUDA
//     kernel on the backend's stream.
//   - eigenpairs() copies results to the host; eigenpairs_device() leaves
//     the eigenvectors on the GPU for downstream device work.
//
// Build: configure with -DARNOLDI_USE_CUDA=ON -DARNOLDI_BUILD_EXAMPLES=ON.

#include <arnoldi/arnoldi.hpp>
#include <arnoldi/cuda.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

// y = A x for the diagonal operator A(i,i) = i+1. The spectrum {1,2,...,n}
// is perfectly separated, so IRA converges fast at any n — a clean showcase
// of a large GPU matvec with exactly-known eigenvalues. (The tridiagonal
// Laplacian is clustered at both spectral ends for large n and would need
// shift-invert to target either end quickly — out of scope for a demo.)
__global__ void diag_matvec(int n, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = static_cast<double>(i + 1) * x[i];
}

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        std::printf("no CUDA device available\n");
        return 0;
    }

    const int n   = 1 << 18;  // 8192 — a non-trivial GPU matvec
    const int nev = 5;
    const int ncv = 8 * nev;  // larger Krylov subspace: the relative gap of
                              // the top eigenvalues is ~1/n, so a bigger ncv
                              // keeps IRA convergence fast at large n

    // "LM": the nev largest eigenvalues, i.e. n, n-1, ..., n-nev+1.
    arnoldi::Arnoldi<arnoldi::Kind::Sym, double, arnoldi::SerialComm,
                     arnoldi::CudaBackend> solver("I", n, "LM", nev, ncv);
    solver.tol(1e-10).maxiter(1000);

    cudaStream_t stream  = solver.backend().stream();
    const int    threads = 256;
    const int    blocks  = (n + threads - 1) / threads;

    // x, y are device pointers (workspace lives on the GPU). Launch on the
    // backend's stream so the surrounding cuBLAS calls order correctly.
    auto t0 = std::chrono::steady_clock::now();
    solver.solve([&](const double* x, double* y) {
        diag_matvec<<<blocks, threads, 0, stream>>>(n, x, y);
    });
    cudaStreamSynchronize(stream);  // ensure all device work is finished
    auto t1     = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    if (!solver.converged()) {
        std::printf("CUDA saupd failed, info = %d\n", solver.info());
        return 1;
    }

    auto r = solver.eigenpairs(false);  // values only (host)
    // eigenpairs(false) returns the wanted set in which-selection order, not
    // sorted by value — sort descending for a clean presentation.
    std::sort(r.values.begin(), r.values.end(), std::greater<double>());
    std::printf("Largest %d eigenvalues of diag(1..%d), CUDA backend:\n", nev, n);
    for (int i = 0; i < nev; ++i) {
        double exact = static_cast<double>(n - i);  // n, n-1, ..., n-nev+1
        std::printf("  lambda[%d] = %.6f   (exact %.0f, |err| = %.2e)\n",
                    i, r.values[i], exact, std::fabs(r.values[i] - exact));
    }
    std::printf("iterations=%d, OP applies=%d\n",
                solver.num_iterations(), solver.num_op_applies());
    std::printf("solve() wall time: %.4f s\n", secs);

    // Eigenvectors can also be kept on the device for further GPU work.
    auto dr = solver.eigenpairs_device(true);
    std::printf("eigenpairs_device: %d eigenvalues on host, "
                "%zu-element eigenvector matrix left on device\n",
                (int)dr.values.size(), (size_t)(n) * nev);
    return 0;
}
