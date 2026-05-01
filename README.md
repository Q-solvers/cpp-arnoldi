![ci workflow](https://github.com/Q-Solvers/cpp-arnoldi/actions/workflows/ci.yml/badge.svg)

# cpp-arnoldi (C++17)

Header-only API for a C++ port of ARPACK’s symmetric and nonsymmetric eigenvalue drivers (`*aupd` / `*eupd`), with additional complex Hermitian support in the symmetric path.

## Requirements

- C++17 compiler
- BLAS and LAPACK (Fortran calling convention, `_` suffix)

## Build (library interface + examples)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/examples/example_symmetric_laplacian
./build/examples/arpack_cpp_examples dnsimp   # Fortran EXAMPLES catalog (see examples/README.md)
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

## Layout

- `include/arnoldi/arnoldi.hpp` — include this
- `include/arnoldi/detail/` — implementation headers
- `examples/` — `example_symmetric_laplacian`, plus `arpack_cpp_examples` (Fortran `EXAMPLES/` name catalog)
- `tools/gen_fortran_examples.py` — optional; regenerates `examples/generated/` when upstream Fortran examples change

## Origin

Derived from the [arpack-ng](https://github.com/opencollab/arpack-ng).

## License

Same as upstream ARPACK / arpack-ng (BSD-style).
