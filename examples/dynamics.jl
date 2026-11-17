# Populations |i⟩⟨i|(t) from the exact propagator vs the SCI eigen-expansion with R modes.
#
#   julia --project=examples examples/dynamics.jl [N=3] [R=10] [dyad|pauli]
include("common.jl")

N = parse(Int, get(ARGS, 1, "3"))
R = parse(Int, get(ARGS, 2, "10"))
basis = parse_basis(get(ARGS, 3, "dyad"))

L = model(N)
ρ0 = Dyad(N, 0, 0)
ts = range(0, 5, length=201)

res = selected_ci(L, ρ0, R; basis=basis)
pops_exact = reduce(hcat, [real(propagate_exact(L, Dyad(N, i, i), ρ0, ts)) for i in 0:2^N-1])
pops_sci = reduce(hcat, [real(propagate(res, Dyad(N, i, i), ρ0, ts)) for i in 0:2^N-1])
@printf("\n max |population error| = %.3e  (dim P = %d, R = %d)\n", maximum(abs, pops_sci - pops_exact), length(res), R)

labels = permutedims(["|" * bitstring(i)[end-N+1:end] * "⟩" for i in 0:2^N-1])
lims = extrema(vcat(pops_exact, pops_sci)) .+ (-0.03, 0.03)
common = (xlabel="time", ylims=lims)
p1 = plot(ts, pops_exact; title="Exact", ylabel="population", legend=false, common...)
p2 = plot(ts, pops_sci; title="SCI  (dim P = $(length(res)), R = $R)", label=labels,
          legend=:outerright, common...)
p = plot(p1, p2, layout=grid(1, 2, widths=[0.43, 0.57]), size=(1050, 400))
savefig_both(p, "dynamics_N$(N)_R$(R)_$(basis_name(basis))")
