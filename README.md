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
| `src/sci.jl` | `selected_ci` (amplitude, first-order or PT2 selection, optionally two-sided; per-iteration callback) |
| `src/lowdin.jl` | `lowdin_correction` with intruder-state diagnostics |
| `src/dynamics.jl` | `propagate`, `propagate_exact` |
| `test/` | `julia --project -e 'using Pkg; Pkg.test()'` (~20 s) |
| `examples/` | plot scripts with their own environment (Plots), figures go to `examples/figures/` |
| `apps/` | larger runs (own environment): `heis_1d.jl` (N = 8 SCI), `heis_1d_exact.jl` (Arnoldi reference), `spin_boson/` |

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/lowdin_convergence.jl 4 dyad
julia --project=examples examples/dynamics.jl 3 10 pauli
julia --project=examples examples/spectrum.jl 3 10
julia --project=examples examples/westhoff_bose_hubbard.jl   # Figs. 2(b), 4 of Westhoff et al., PRX 16, 041020
julia --project=examples examples/selection_comparison.jl    # error vs dim P: right-only, two-sided, PT2 selection
```

Selection rules: by default a key enters P when the right eigenvectors reach it (`|(L v)_q| > ϵ_add`).
In a non-normal L the left eigenvectors can live elsewhere: `two_sided=true` also scores them, and
`selection=:pt2` scores a key by the Löwdin term it would add, `|(w L)_q (L v)_q / (λ − L_qq)|`
(a rate, so `ϵ_add ≈ 1e-4`).

Models that are awkward as Pauli strings (bosons) use `MatrixLindbladian`: H and jumps as sparse
matrices over a chosen basis, SCI keys are `(ket, bra)` pairs. `dissipative_bose_hubbard(L, N;
basis=:momentum)` builds the driven Bose-Hubbard chain in the standing-wave Fock basis, where its
slow modes are compact.
