![ci workflow](https://github.com/Q-Solvers/cpp-arnoldi/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/github/Q-solvers/cpp-arnoldi/graph/badge.svg?token=M8DQCOO8I2)](https://codecov.io/github/Q-solvers/cpp-arnoldi)

# cpp-arnoldi (C++17)

Header-only API for a C++ port of ARPACK’s symmetric and nonsymmetric eigenvalue drivers (`*aupd` / `*eupd`), with additional complex Hermitian support in the symmetric path.

## Requirements

- C++17 compiler
- BLAS and LAPACK (Fortran calling convention, `_` suffix)
- *(optional, GPU backend)* CUDA Toolkit with cuBLAS — see [GPU (CUDA) backend](#gpu-cuda-backend)
- *(optional, tests)* none to install: Catch2 v3 is fetched automatically via CMake `FetchContent` (needs network on first configure)

## Build (library interface + examples)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/examples/example_symmetric_laplacian
./build/examples/arpack_cpp_examples dnsimp   # Fortran EXAMPLES catalog (see examples/README.md)
```

Options:

| Option | Default | Effect |
| --- | --- | --- |
| `ARNOLDI_BUILD_EXAMPLES` | OFF | build the `examples/` programs |
| `ARNOLDI_BUILD_TESTS` | OFF | build the Catch2 test suite (`ctest`) |
| `ARNOLDI_USE_CUDA` | OFF | enable the CUDA + cuBLAS backend (Sym / Herm) |

```bash
# tests
cmake -S . -B build -DARNOLDI_BUILD_TESTS=ON
cmake --build build && ctest --test-dir build
```

## Use in your project

```cmake
add_subdirectory(cpparnoldi)   # or find_package after install
target_link_libraries(your_target PRIVATE arpack::callback)
```

```cpp
#include <arnoldi/detail.hpp>
// arnoldi::detail::naupd, neupd, saupd, seupd, ...
```

## GPU (CUDA) backend

`Arnoldi<K, Scalar, Comm, Backend>` is templated on a backend. The default
`CpuBackend` runs every length-`n` BLAS on the host (unchanged behaviour).
`CudaBackend` keeps the Lanczos workspace (`resid`, `v`, `workd`) in device
memory and routes those operations through cuBLAS, so the **matvec runs
entirely on the GPU**. Supported for `Kind::Sym` and `Kind::Herm` (Nonsym on
device is rejected at compile time — no cuSOLVER analogue for its real-Schur
path). The small implicitly-restarted shift machinery (`ncv²`-sized `h`,
`q`, `workl`) stays host-resident by design.

Enable it:

```bash
cmake -S . -B build -DARNOLDI_USE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=<your_sm>          # e.g. 61 for Quadro P1000
cmake --build build
```

`cuda.hpp` is plain C++ calling the cuBLAS C API — only the `.cu`
tests/examples need `nvcc`. If your system compiler is newer than what your
`nvcc` supports (e.g. CUDA 12.1 + GCC ≥ 13), point `nvcc` at an older host
compiler:

```bash
cmake -S . -B build -DARNOLDI_USE_CUDA=ON \
      -DCMAKE_CUDA_HOST_COMPILER=/path/to/g++-12 \
      -DCMAKE_CUDA_ARCHITECTURES=61
```

Usage — the matvec callback receives **device** pointers and should launch on
the backend's stream:

```cpp
#include <arnoldi/arnoldi.hpp>
#include <arnoldi/cuda.hpp>

arnoldi::Arnoldi<arnoldi::Kind::Sym, double,
                 arnoldi::SerialComm, arnoldi::CudaBackend>
    s("I", n, "LM", nev, ncv);
s.tol(1e-10).maxiter(1000);

cudaStream_t stream = s.backend().stream();
s.solve([&](const double* d_x, double* d_y) {       // device pointers
    my_matvec_kernel<<<blocks, threads, 0, stream>>>(n, d_x, d_y);
});

auto r  = s.eigenpairs();          // values + vectors copied to host
auto dr = s.eigenpairs_device();   // eigenvectors left on the GPU
```

Also available under a device backend: `s.initial_resid_device(devptr)` to
seed from a device-resident vector, and `s.backend().handle()` for the cuBLAS
handle. See `examples/arnoldi_sym_cuda.cu` (and `arnoldi_sym_cpu.cpp` for the
matching CPU baseline / benchmark).

## Layout

- `include/arnoldi/arnoldi.hpp` — include this
- `include/arnoldi/cuda.hpp` — opt-in CUDA backend (`-DARNOLDI_USE_CUDA=ON`)
- `include/arnoldi/detail/` — implementation headers
- `examples/` — `example_symmetric_laplacian`, `arnoldi_sym_cuda` / `arnoldi_sym_cpu`, plus `arpack_cpp_examples` (Fortran `EXAMPLES/` name catalog)
- `tests/` — Catch2 suite (`ctest`), incl. `test_solver_cuda` when CUDA is on

## Origin

Derived from the [arpack-ng](https://github.com/opencollab/arpack-ng).

## License

Same as upstream ARPACK / arpack-ng (BSD-style).
