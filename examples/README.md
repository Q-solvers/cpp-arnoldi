# Examples (Fortran `EXAMPLES/` catalog)

This directory mirrors **every serial** program under `EXAMPLES/` in upstream [arpack-ng](https://github.com/opencollab/arpack-ng) (78 Fortran files). Each name can be run as:

```bash
./arpack_cpp_examples <name>
```

Regenerate `generated/example_names.inc` and `generated/stubs.cpp` when upstream adds or renames Fortran examples — from a **full arpack-ng** checkout with `EXAMPLES/` next to this package:

```bash
cd arpack_callback
python3 tools/gen_fortran_examples.py
```

If you only have this repository, use the committed `generated/` files.

## Coverage

| Status | Meaning |
|--------|--------|
| **Ported** | Implemented in `ported.cpp` or `ported_extended.cpp`; matches the Fortran driver settings and operator (or a documented twin). |
| **Stub** | `generated/stubs.cpp` — prints a message and exits `79`. |
| **N/A** | Not representable with the current callback API (see below). |

### Fully ported

| Name | Fortran source | Notes |
|------|----------------|--------|
| `dnsimp` | `EXAMPLES/SIMPLE/dnsimp.f` | Nonsymmetric, `LM`, double, 2D convection–diffusion (ρ=0). |
| `snsimp` | `EXAMPLES/SIMPLE/snsimp.f` | Same, **float**. |
| `dssimp` | `EXAMPLES/SIMPLE/dssimp.f` | Symmetric 2D Laplacian, `LM`, double. |
| `sssimp` | `EXAMPLES/SIMPLE/sssimp.f` | Same, **float**. |
| `dndrv1` | `EXAMPLES/NONSYM/dndrv1.f` | Nonsymmetric, `SM`, double. |
| `dndrv2`–`dndrv4` | `EXAMPLES/NONSYM/dndrv*.f` | Tridiagonal FE / conv–diff; `dgttrf`/`dgttrs` or `dpttrf`/`dpttrs` as in Fortran. |
| `sndrv1`–`sndrv4` | `EXAMPLES/NONSYM/sndrv*.f` | **Float** counterparts. |
| `dsdrv1`–`dsdrv6` | `EXAMPLES/SYM/dsdrv*.f` | Symmetric drivers: `dsdrv1` uses `SM` (2D Laplacian); `dsdrv2`–`dsdrv6` use 1D FE / Laplacian and modes 2–5 as in Fortran. |
| `ssdrv1`–`ssdrv6` | `EXAMPLES/SYM/ssdrv*.f` | **Float** counterparts. |
| `dsvd`, `ssvd` | `EXAMPLES/SVD/dsvd.f`, `ssvd.f` | Largest singular values via `A'*A` matvec chain. |

Kernels and LAPACK wrappers: `kernels.hpp`, `lapack_extra.hpp`.

### Stubs (exit 79)

Remaining names include: **band** family (`*band`, `*bdr*`), **complex nonsymmetric** drivers, and **nonsymmetric examples that require complex tridiagonal solves** (`dndrv5`, `dndrv6`, `sndrv5`, `sndrv6` — Fortran uses `zgttrf`/`zgttrs`).

### Parallel (not in this package)

`PARPACK/EXAMPLES/MPI/*.f` and `PARPACK/EXAMPLES/BLACS/*.f` are not included here (MPI/BLACS not in this library).

### Complex nonsymmetric (`cnaupd` / `znaupd`)

Programs such as `znsimp`, `cnsimp`, `cndrv*`, `zndrv*` need a complex nonsymmetric callback path not present yet.
