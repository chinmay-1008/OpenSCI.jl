# OpenSCI

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall-vt.github.io/OpenSCI.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nmayhall-vt.github.io/OpenSCI.jl/dev/)
[![Build Status](https://github.com/nmayhall-vt/OpenSCI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nmayhall-vt/OpenSCI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nmayhall-vt/OpenSCI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/OpenSCI.jl)

Selected-CI for the slow eigenmodes of a Lindbladian, a second-order Löwdin correction for
the truncated space, and eigen-expansion dynamics. One implementation works in either
operator basis: dyads `|k⟩⟨b|` (`DyadBasis`) or Pauli strings (`PauliBasis`), built on
[PauliOperators.jl](https://github.com/nmayhall/PauliOperators.jl) ≥ 3.1.

```julia
using OpenSCI, PauliOperators
N = 4
L = Lindbladian(N)
add_hamiltonian!(L, heisenberg_1D(N, 1.1, 1.2, 1.3))
add_channel_dephasing!(L, 0.1)
add_channel_depolarizing!(L, 0.1)

res = selected_ci(L, Dyad(N, 0, 0), 2)                 # basis = PauliBasis also works
lowdin_correction(L, res).λ                            # second-order corrected eigenvalues
propagate(res, Pauli(N, Z=[2, 3]), Dyad(N, 0, 0), 0:0.1:5)
```

## Layout

| file | contents |
|---|---|
| `src/lindbladian.jl` | `Lindbladian`, noise channels, `SuperOpTerms` (L as Pauli sandwiches), `Matrix(L, basis)` |
| `src/basis.jl` | basis conversion, `subspace_matrix` (L_PP, L_PQ, …) |
| `src/eigen.jl` | top-R eigenpairs with biorthogonal left eigenvectors (`W*V = I`, no `pinv`) |
| `src/sci.jl` | `selected_ci` (amplitude or first-order selection, per-iteration callback) |
| `src/lowdin.jl` | `lowdin_correction` with intruder-state diagnostics |
| `src/dynamics.jl` | `propagate`, `propagate_exact` |
| `test/` | `julia --project -e 'using Pkg; Pkg.test()'` (~20 s) |
| `examples/` | plot scripts with their own environment (Plots), figures go to `examples/figures/` |

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/lowdin_convergence.jl 4 dyad
julia --project=examples examples/dynamics.jl 3 10 pauli
julia --project=examples examples/spectrum.jl 3 10
```
