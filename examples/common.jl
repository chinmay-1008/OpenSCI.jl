# Shared setup for the example scripts: the model and a consistent plot style.
using OpenSCI, PauliOperators, LinearAlgebra, Printf
using Plots, Plots.PlotMeasures

"""Heisenberg ring with dephasing + depolarizing noise (the model used throughout the paper plots)."""
function model(N; J=(1.1, 1.2, 1.3), dephasing=0.1, depolarizing=0.1)
    L = Lindbladian(N)
    add_hamiltonian!(L, heisenberg_1D(N, J...))
    add_channel_dephasing!(L, dephasing)
    add_channel_depolarizing!(L, depolarizing)
    return L
end

"""Exact spectrum sorted by decreasing real part (dense, 4^N × 4^N)."""
exact_spectrum(L) = sort(eigvals(Matrix(L)), by=z -> (-real(z), -imag(z)))

parse_basis(s) = lowercase(s) in ("pauli", "p") ? PauliBasis : DyadBasis
basis_name(b) = b === PauliBasis ? "pauli" : "dyad"

const FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

default(
    fontfamily="Computer Modern", framestyle=:box, grid=true, gridalpha=0.25,
    linewidth=2, markerstrokewidth=0.5, dpi=300,
    titlefontsize=11, guidefontsize=10, tickfontsize=8, legendfontsize=8,
    left_margin=4mm, bottom_margin=4mm, right_margin=2mm, top_margin=2mm,
    palette=:tab10,
)

savefig_both(p, name) = (savefig(p, joinpath(FIGDIR, name * ".png")); savefig(p, joinpath(FIGDIR, name * ".pdf"));
                         println(" wrote ", joinpath(FIGDIR, name * ".{png,pdf}")))
