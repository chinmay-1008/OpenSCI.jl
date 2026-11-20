# OpenSCI: plan and lab notebook

The goal: noisy open-system dynamics from the slow eigenmodes of the Lindbladian. Selected CI (SCI) keeps the space small, and a second-order Löwdin correction recovers what the truncation loses.

This file records what we have done, how to reproduce every result on a fresh machine, what we learned, and what comes next.

---

## 1. Where things are

| what | where |
|---|---|
| my fork | https://github.com/chinmay-1008/OpenSCI.jl |
| PI's repo (the fork's parent) | https://github.com/nmayhall-vt/OpenSCI.jl |
| working branch | `rewrite` (latest commit `6e885f3`, pushed) |
| old code | `paulis_v2.0` (PI's branch, plus my port in commit `37e1706`) |
| `PauliOperators` 3.1 | https://github.com/nmayhall/PauliOperators.jl, installed through the lab registry (see §2) |
| paper we recreate | Westhoff, Moroder, Schollwöck, Paeckel, *Tensor-Network Framework for Lindbladian Spectra and Steady States*, PRX 16, 041020 (2026) |

Commits on `rewrite`, oldest first:

1. `37e1706`: port to PauliOperators 3.1, fix the Löwdin left eigenvectors, trim dependencies (made on `paulis_v2.0`, before the branch).
2. `1c8e90a`: rewrite; one generic SCI / Löwdin / dynamics core for the dyad and Pauli bases.
3. `3827199`: port `apps/` to the new API; multi-seed `selected_ci`; `sparse(L)`.
4. `6e885f3`: sparse-matrix backend, Bose-Hubbard model, `keep_seeds`, two-sided and PT2 selection, Westhoff recreation, selection comparison.

---

## 2. Set up on a new machine

Julia: I used 1.12.7 (through juliaup); the package needs 1.11 or newer.

```bash
git clone https://github.com/chinmay-1008/OpenSCI.jl.git
cd OpenSCI.jl
git checkout rewrite
git remote add upstream https://github.com/nmayhall-vt/OpenSCI.jl.git
```

PauliOperators 3.1 is not in Julia's General registry. Add the lab registry first, once per machine. Its listed address is `git@github.com:mayhallgroup/MayhallJuliaRegistry.git`, which needs an SSH key on GitHub; the https form below may also work.

```bash
julia -e 'using Pkg; Pkg.Registry.add(url="https://github.com/mayhallgroup/MayhallJuliaRegistry")'
```

Then install the three environments. Each has a committed `Manifest.toml`, so you get the same package versions:

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=apps -e 'using Pkg; Pkg.instantiate()'
```

The tests should report **132 passed** in about 25 s.

---

## 3. Reproduce the results

All figures go to `examples/figures/` as `.png` and `.pdf`. That folder is **not in git**, so a new machine recomputes everything.

The slow scripts cache their runs in that same folder as `.jls` files. A second run only redraws, in seconds.

**Stale caches:** a cache doesn't know when the code changed. After changing anything in `src/`, delete the matching `examples/figures/bh_*.jls` files before rerunning.

| result | command | output | notes |
|---|---|---|---|
| tests | `julia --project -e 'using Pkg; Pkg.test()'` | 132 pass | ~25 s |
| Löwdin convergence (spin chain) | `julia --project=examples examples/lowdin_convergence.jl 4 dyad` (also `pauli`) | `lowdin_convergence_N4_{dyad,pauli}` | |
| dynamics vs exact | `julia --project=examples examples/dynamics.jl 3 10 dyad` (also `pauli`) | `dynamics_N3_R10_*` | |
| spectrum vs SCI | `julia --project=examples examples/spectrum.jl 3 10 dyad` (also `pauli`) | `spectrum_N3_R10_*` | |
| Westhoff Figs. 2(b) and 4 | `julia --project=examples examples/westhoff_bose_hubbard.jl` | `westhoff_fig2b_L5_N4`, `westhoff_fig4_L5_N4` | caches `bh_exact`, `bh_krylov_baseline`, `bh_sci`; the first run takes tens of minutes |
| selection rules, error vs dim P | see below | `selection_comparison_L5_N4` | 13 runs, ~25 min with three processes in parallel |
| noisy Heisenberg ring, N = 8 | `julia --project=apps apps/heis_1d.jl 8 dyad 4` | printed | `max_iter = 4` as in the old app |
| exact reference (Arnoldi) | `julia --project=apps apps/heis_1d_exact.jl 6` | printed | |
| spin–boson toy problems | `julia --project=apps apps/spin_boson/spin_boson.jl` | PDFs in `apps/spin_boson/` | |

To run the selection comparison in parallel, start the three methods' runs as separate processes, then draw the figure once they finish:

```bash
julia --project=examples examples/selection_comparison.jl right &
julia --project=examples examples/selection_comparison.jl two_sided &
julia --project=examples examples/selection_comparison.jl pt2 &
wait
julia --project=examples examples/selection_comparison.jl        # loads all cached runs, draws the figure
```

`selection_comparison.jl plot` draws only the runs that are already cached, which is handy while the others are still going.

The right-only run at ϵ = 1e-2 is the slowest, about 12 min. It never converges, so it runs all 20 iterations.

---

## 4. Results to check against

### 4.1 Westhoff et al., dissipative Bose-Hubbard

The setup: L = 5 sites, N = 4 bosons, J = U = κ = 1. That gives 70 Fock states and a Liouville dimension of 4900, in the momentum basis.

The exact values are λ₂ = −0.8409 + 0.8135i (the paper gives −0.84 + 0.81i), λ₃ = −1.3485, λ₄ = −1.4484 + 1.4497i and λ₅ = −2.0162 + 1.9712i.

The table gives the deviation |λⱼ − λ_exact|:

| method | dim P | λ₂ | λ₃ | λ₄ | λ₅ |
|---|---|---|---|---|---|
| real-time Krylov (geometric mean over 50 random states) | — | 0.12 | 0.14 | 0.21 | 1.6 |
| SCI | 2014 | 2.1e-2 | 5.7e-3 | 3.1e-2 | 1.0e-2 |
| SCI + Löwdin | 2014 | 5.6e-3 | 2.0e-3 | 8.2e-3 | 6.2e-3 |
| SCI | 3101 | 1.2e-2 | 3.5e-3 | 9.9e-3 | 3.1e-3 |
| SCI + Löwdin | 3101 | 1.9e-3 | 5.8e-4 | 2.2e-3 | 1.5e-3 |
| SCI | 4046 | 8.6e-4 | 3.7e-4 | 2.0e-3 | 2.2e-3 |
| SCI + Löwdin | 4046 | 4.3e-4 | 2.0e-4 | 5.5e-4 | 7.7e-4 |

The steady state λ₁ is exact to about 1e-13.

### 4.2 Selection rules (same model)

How a key q enters P, with ϵ = `ϵ_add`:

- **right only** (the default): |(L v)_q| > ϵ.
- **two-sided** (`two_sided=true`): the left eigenvectors are scored too, normalized to unit length, and a key is clipped only if it is small in both.
- **PT2** (`selection=:pt2`): |(wL)_q (Lv)_q / (λ − L_qq)| > ϵ, which is the Löwdin term q would add.

Every run uses `R = 10`, `max_iter = 20`, and `ϵ_clip = ϵ/10` (1e-4 for PT2). Ranges are best to worst over λ₂ … λ₅. "Left" and "right" are the eigenvector infidelities, 1 − |⟨x|x_exact⟩|.

| rule | ϵ | dim P | iterations | SCI | SCI + Löwdin | right | left |
|---|---|---|---|---|---|---|---|
| right | 1e-2 | 2014 | 20, not converged | 6e-3 – 3e-2 | 2e-3 – 8e-3 | 7e-5 – 2e-4 | 0.05 – 0.32 |
| right | 3e-3 | 3101 | 20, not converged | 3e-3 – 1.2e-2 | 6e-4 – 2e-3 | 1e-5 – 3e-5 | 0.03 – 0.22 |
| right | 1e-3 | 4046 | 4 | 4e-4 – 2e-3 | 2e-4 – 8e-4 | 4e-7 – 2e-6 | 0.02 – 0.10 |
| right | 3e-4 | 4651 | 20, not converged | 8e-5 – 7e-4 | 5e-5 – 1.3e-4 | 3e-8 – 1.3e-7 | 9e-3 – 6e-2 |
| right | 1e-4 | 4866 | 4 | 9e-6 – 9e-5 | 3e-7 – 7e-6 | 1e-10 – 6e-9 | 1e-3 – 1e-2 |
| two-sided | 0.2 | 1897 | 8 | 1.3e-2 – 0.11 | 1e-3 – 2e-2 | 3e-3 – 1e-2 | 5e-3 – 4e-2 |
| two-sided | 0.1 | 2791 | 7 | 8e-3 – 2e-2 | 2e-3 – 6e-3 | 5e-4 – 2e-3 | 1e-3 – 1e-2 |
| two-sided | 0.05 | 3906 | 6 | 1e-3 – 4e-3 | 1.5e-4 – 6e-4 | 5e-5 – 2.5e-4 | 8e-5 – 1e-3 |
| two-sided | 0.01 | 4840 | 4 | 1.5e-5 – 6e-5 | 2e-7 – 1.3e-5 | 8e-8 – 1e-6 | 1e-7 – 3e-6 |
| PT2 | 1e-2 | 2514 | 6 | 4e-3 – 5e-2 | 1.4e-3 – 2e-2 | 2e-4 – 3e-3 | 8e-4 – 0.40 |
| PT2 | 1e-3 | 3061 | 7 | 1.5e-3 – 6e-3 | 4e-4 – 3.5e-3 | 3e-5 – 8e-5 | 6e-4 – 0.18 |
| PT2 | 1e-4 | 4056 | 9 | 1.1e-4 – 6e-4 | 3e-5 – 1.4e-4 | 5e-7 – 5e-6 | 3e-4 – 3e-2 |
| PT2 | 1e-5 | 4734 | 6 | 4e-6 – 4e-5 | 7e-7 – 1.3e-5 | 3e-9 – 1.5e-7 | 3e-6 – 3e-4 |

What the table says:

- **Best eigenvalues per dyad: PT2.** At dim P ≈ 4050 it is about 5× more accurate than right-only, both before and after Löwdin.
- **Best left eigenvectors: two-sided.** At about 4000 dyads they are about 100× better than right-only's. Its right eigenvectors are the worst of the three.
- **Convergence:** right-only flips between two P spaces of nearly equal size, and never converged at 3 of its 5 thresholds. Two-sided and PT2 converged in every run.
- **No rule compresses much here.** An error of about 1e-4 needs about 80% of the space with any rule.

---

## 5. What we learned

- **Basis matters most.** In the site Fock basis the slow modes are delocalized: 99% of each mode's weight needs about 75% of all dyads. In the standing-wave (momentum) basis that drops to 43–97 dyads.
- **Truncation pushes eigenvalues left** (more negative real part). The external space Q acts like an absorbing boundary.
- **Never clip the seeds.** Clipping population dyads breaks trace conservation inside P; the steady state then drifted to −1.17. Hence `keep_seeds=true`.
- **Symmetry sectors need their own seeds.** λ₂ is odd under reflection, and population seeds are all even, so SCI never reaches it. Coherences with the condensate state |N,0,…,0⟩ fix this.
- **Left eigenvectors live elsewhere.** With right-only selection, the exact right eigenvectors keep about 1e-7 of their weight outside P, but the left eigenvectors keep 4–19%. This limits Löwdin and the w·ρ₀ coefficients used in dynamics.
- **Pair Löwdin corrections with their own state.** Never re-sort the corrected eigenvalues.
- **Match eigenvalues by value, not by position.** Real eigenvalues from Arnoldi carry an imaginary part of about 1e-8. With an upper-half-plane filter at −1e-8 such a mode got dropped, which shifted every later mode by one and produced fake errors of about 1. The scripts now use −1e-6, or nearest-eigenvalue matching.
- **`pinv(V)` is not a left eigenbasis** for a non-normal L. `biorthogonal_eigen` solves the transpose problem instead; a test shows the difference.
- **Bosons as Pauli strings blow up:** one binary-encoded boson jump operator is about 87,000 Pauli sandwich terms. Hence `MatrixLindbladian`.

---

## 6. Next steps

### Housekeeping

- [ ] Remove `.DS_Store` and `.vscode/settings.json` from the branch (they came in with early commits) and add them to `.gitignore`.
- [ ] Agree with the PI on credit: add myself to `authors` in `Project.toml`, and maybe add a `CITATION.cff`.
- [ ] Open a pull request into the PI's `paulis_v2.0`. Compare page: https://github.com/nmayhall-vt/OpenSCI.jl/compare/paulis_v2.0...chinmay-1008:OpenSCI.jl:rewrite. The description draft is `PR_DESCRIPTION.md` on the desktop only; it is excluded from git there. Commits count on my GitHub graph only once they reach the PI's `main`.
- [ ] Optional: push my local `paulis_v2.0` (one commit ahead of my fork's copy). That commit is already in `rewrite`'s history.

### Research

1. **Stop the right-only oscillation.** Ideas: don't clip keys added in the previous iteration, or stop when P returns to a set seen before and use the union of the two.
2. **Scaling with system size: the central question.** Does dim P / D shrink as L grows? Next sizes: L = 6, N = 4 (126 states, D = 15,876) and L = 7. The exact references from Arnoldi are still feasible there.
3. **Dynamics on Bose-Hubbard.** Run `propagate` against `propagate_exact` and see how much the left-vector quality (two-sided vs right-only) changes ρ(t).
4. **Better Löwdin.** It currently uses only the diagonal of L_QQ (Epstein–Nesbet style). Try off-diagonal L_QQ couplings, or a self-consistent Brillouin–Wigner iteration.
5. **Better bases and symmetry blocks.** Split by reflection parity to halve the space. Try a basis built from the steady state's eigenvectors.
6. **Combine the rules.** For example, PT2 for adding keys with two-sided clipping, or tune PT2's `ϵ_clip`; all PT2 runs here used 1e-4.
7. **Rest of the Westhoff paper.**
   - Fig. 8(a), the steady-state correlation ⟨b₁†b₅⟩: proposed, not started.
   - Fig. 4(b)'s × markers, the real-time states at T = 4 and T = 8: not reproduced.
   - The Fig. 2(b) inset at L = 20: out of reach for exact references.
8. **Speed.** P above `dense_max = 3000` switches to Arnoldi, and the slowest runs are the non-converging ones. Profile `external_action` and `subspace_matrix` on the Bose-Hubbard runs.
