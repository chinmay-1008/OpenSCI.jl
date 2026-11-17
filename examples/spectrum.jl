# Full Lindbladian spectrum vs the SCI eigenvalues.
#
#   julia --project=examples examples/spectrum.jl [N=3] [R=10] [dyad|pauli]
include("common.jl")

N = parse(Int, get(ARGS, 1, "3"))
R = parse(Int, get(ARGS, 2, "10"))
basis = parse_basis(get(ARGS, 3, "dyad"))

L = model(N)
λex = exact_spectrum(L)
res = selected_ci(L, Dyad(N, 0, 0), R; basis=basis)

p1 = scatter(real(λex), imag(λex), title="Full diagonalization ($(4^N) eigenvalues)",
             xlabel="Re λ", ylabel="Im λ", ms=2.5, color=:gray40, legend=false)
p2 = scatter(real(λex), imag(λex), ms=2.5, color=:gray80, label="exact",
             title="SCI  (dim P = $(length(res)), R = $R)", xlabel="Re λ")
scatter!(p2, real(res.λ), imag(res.λ), ms=5, color=4, marker=:diamond, label="SCI")
p = plot(p1, p2, layout=(1, 2), size=(1000, 400), link=:all)
savefig_both(p, "spectrum_N$(N)_R$(R)_$(basis_name(basis))")
