# Error of the slowest-decaying eigenvalue vs SCI subspace size, with and without the
# second-order Löwdin correction, for both selection rules.
#
#   julia --project=examples examples/lowdin_convergence.jl [N=4] [dyad|pauli]
include("common.jl")

N = parse(Int, get(ARGS, 1, "4"))
basis = parse_basis(get(ARGS, 2, "dyad"))
mode = 2                                   # λ[1] = steady state, λ[2] = slowest decay
intruder_tol = 1e-3

L = model(N)
λex = exact_spectrum(L)
target = λex[mode]
err(λ) = min(abs(λ - target), abs(λ - conj(target)))
@printf("N = %d, %s basis, exact λ[%d] = %.10f %+.10fi\n", N, nameof(basis), mode, real(target), imag(target))

runs = Dict{Symbol,Any}()
for selection in (:amplitude, :perturbative)
    rows = NamedTuple[]
    selected_ci(L, Dyad(N, 0, 0), 2; basis=basis, selection=selection, max_iter=12, verbose=false,
        callback=res -> begin
            lw = lowdin_correction(L, res)
            push!(rows, (dim=length(res), sci=err(res.λ[mode]), low=err(lw.λ[mode]),
                         intruder=lw.min_denominator[mode] < intruder_tol))
        end)
    runs[selection] = rows
    println("\n selection = :$selection")
    @printf(" %4s %8s %14s %14s\n", "iter", "dim P", "|err SCI|", "|err Löwdin|")
    for (i, r) in enumerate(rows)
        @printf(" %4d %8d %14.3e %14.3e%s\n", i, r.dim, r.sci, r.low, r.intruder ? "  (intruder)" : "")
    end
end

# Intruder-state blowups (up to ~1e14) would squash the useful range, so cap the axis and
# draw anything above it on the top edge.
ymax = 1e2
show_err(x) = clamp(x, 1e-16, ymax)        # floor: exact zeros would vanish on a log axis
p = plot(xlabel="SCI subspace size  dim P", ylabel="|λ − λ(exact)|", yscale=:log10, legend=:bottomleft,
         title="Slowest decay mode, N = $N ($(basis_name(basis)) basis)", size=(640, 440),
         ylims=(0.3 * minimum(show_err(x.low) for r in values(runs) for x in r), 3ymax))
hline!(p, [ymax], color=:gray60, ls=:dot, lw=1, label="")
for (selection, ls, name) in ((:amplitude, :solid, ""), (:perturbative, :dash, ", 1st-order select"))
    r = runs[selection]
    d = [x.dim for x in r]
    plot!(p, d, show_err.([x.sci for x in r]), color=1, ls=ls, marker=:circle, ms=4, label="SCI" * name)
    plot!(p, d, show_err.([x.low for x in r]), color=4, ls=ls, marker=:diamond, ms=4, label="SCI + Löwdin" * name)
    bad = findall(x -> x.intruder, r)
    isempty(bad) || scatter!(p, d[bad], show_err.([r[i].low for i in bad]), marker=:xcross, ms=8,
                             color=:black, msw=2,
                             label=selection === :amplitude ? "intruder: |λ − L_qq| < $intruder_tol" : "")
end
annotate!(p, [(minimum(x.dim for r in values(runs) for x in r), 1.8ymax,
               text(" ≥ 100 (off scale)", 8, :left, :gray40, "Computer Modern"))])
savefig_both(p, "lowdin_convergence_N$(N)_$(basis_name(basis))")
