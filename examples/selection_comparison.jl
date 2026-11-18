# Error vs dim P for three SCI selection rules on the dissipative Bose-Hubbard chain of
# Westhoff et al., PRX 16, 041020 (L = 5, N = 4, J = U = κ = 1, momentum basis, Liouville
# dimension 4900). An external key q enters P when its score exceeds ϵ_add:
#   right only : |(L v)_q|                          (the default)
#   two-sided  : max(|(L v)_q|, |(ŵ L)_q|)          ŵ = left eigenvector, unit norm
#   PT2        : |(w L)_q (L v)_q / (λ − L_qq)|     the Löwdin term q would contribute
# Errors are over the slow modes λ₂ … λ₅ (geometric mean, band = best to worst mode).
#
#   julia --project=examples examples/selection_comparison.jl          # all runs (cached) + plot
#   julia --project=examples examples/selection_comparison.jl pt2      # one method's runs only, no plot
include("common.jl")
using SparseArrays, Serialization
using OpenSCI: biorthogonal_eigen

const Lc, Nb, NJ, R = 5, 4, 5, 10
const METHODS = [
    (key="right", label="right only", selection=:amplitude, two_sided=false, eps=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4]),
    (key="two_sided", label="two-sided", selection=:amplitude, two_sided=true, eps=[2e-1, 1e-1, 5e-2, 1e-2]),
    (key="pt2", label="PT2", selection=:pt2, two_sided=false, eps=[1e-2, 1e-3, 1e-4, 1e-5]),
]

Λ, states = dissipative_bose_hubbard(Lc, Nb; J=1, U=1, κ=1, basis=:momentum)
d = length(states)
D = d^2
cached(f, name) = (p = joinpath(FIGDIR, name * ".jls"); isfile(p) ? deserialize(p) : (x = f(); serialize(p, x); x))

ex = cached("bh_exact") do                         # shared with westhoff_bose_hubbard.jl
    λ, V, W = biorthogonal_eigen(sparse(Λ), 16; dense_max=0)
    (λ=λ, V=V, W=W)
end
# upper half plane by decreasing real part; real eigenvalues from Arnoldi carry ~1e-8 imaginary noise
upper_idx(λ) = filter(i -> imag(λ[i]) > -1e-6, sortperm(λ, by=z -> (-real(z), -imag(z))))
jex = upper_idx(ex.λ)[1:NJ]
λex = ex.λ[jex]

c = findfirst(==([Nb; zeros(Int, Lc - 1)]), states)    # condensate |N,0,…,0⟩
seeds = unique([[(i, i) for i in 1:d]; [(c, m) for m in 1:d]; [(m, c) for m in 1:d]])
embed(res, v) = (x = zeros(ComplexF64, D); for (k, (a, b)) in enumerate(res.basis); x[a+(b-1)*d] = v[k]; end; x)
infid(a, b) = 1 - abs(dot(a, b)) / (norm(a) * norm(b))

function run(m, ϵ)
    cached("bh_sel_$(m.key)_$(ϵ)") do
        t = @elapsed res = selected_ci(Λ, seeds, R; selection=m.selection, two_sided=m.two_sided, ϵ_add=ϵ,
                                       ϵ_clip=(m.selection === :pt2 ? 1e-4 : ϵ / 10), max_iter=20, verbose=false)
        lw = lowdin_correction(Λ, res)
        js = [argmin(abs.(res.λ .- z)) for z in λex]     # nearest SCI eigenvalue to each exact one
        (ϵ=ϵ, dimP=length(res), iteration=res.iteration, converged=res.converged, time=t,
         λerr=[abs(res.λ[js[j]] - λex[j]) for j in 1:NJ],
         λerr_low=[abs(lw.λ[js[j]] - λex[j]) for j in 1:NJ],
         r_inf=[infid(embed(res, res.V[:, js[j]]), ex.V[:, jex[j]]) for j in 1:NJ],
         l_inf=[infid(embed(res, res.W[js[j], :]), ex.W[jex[j], :]) for j in 1:NJ])
    end
end

fmt(v) = join([@sprintf("%.1e", x) for x in v[2:end]], " ")
plot_only = ARGS == ["plot"]                       # plot whatever is cached, run nothing
todo = isempty(ARGS) || plot_only ? METHODS : filter(m -> m.key in ARGS, METHODS)
done(m, ϵ) = isfile(joinpath(FIGDIR, "bh_sel_$(m.key)_$(ϵ).jls"))
results = Dict(m.key => map(filter(ϵ -> !plot_only || done(m, ϵ), m.eps)) do ϵ
                   r = run(m, ϵ)
                   @printf(" %-10s ϵ=%.0e dim P=%4d it=%2d%s %6.1fs | λ %s | Löwdin %s | r %s | l %s\n", m.key, ϵ,
                           r.dimP, r.iteration, r.converged ? " conv" : "     ", r.time, fmt(r.λerr),
                           fmt(r.λerr_low), fmt(r.r_inf), fmt(r.l_inf))
                   flush(stdout)
                   r
               end for m in todo)
isempty(ARGS) || plot_only || exit()

# ---------------------------------------------------------------- plot
geo(v) = exp(sum(log.(max.(v, 1e-16))) / length(v))
function curve!(p, rs, f; color, label, ls=:solid, marker=:circle, ribbon=true)
    isempty(rs) && return
    rs = sort(rs, by=r -> r.dimP)
    x = [r.dimP for r in rs]
    g = [geo(f(r)[2:NJ]) for r in rs]
    lo = [minimum(f(r)[2:NJ]) for r in rs]
    hi = [maximum(f(r)[2:NJ]) for r in rs]
    ribbon && plot!(p, x, lo, fillrange=hi, fillalpha=0.12, lw=0, color=color, label="")
    plot!(p, x, g, color=color, ls=ls, marker=marker, ms=4, msc=color, label=label)
end

# one y range for all panels: whole decades around every plotted value (all modes, both sides)
allv = [max(x, 1e-16) for rs in values(results) for r in rs
        for v in (r.λerr, r.λerr_low, r.r_inf, r.l_inf) for x in v[2:NJ]]
ydec = floor(Int, log10(minimum(allv))):ceil(Int, log10(maximum(allv)))
panel(title, ylabel) = plot(title=title, ylabel=ylabel, xlabel="dim P  (of $D)", yscale=:log10,
                            ylims=10.0 .^ extrema(ydec), yticks=10.0 .^ ydec, xlims=(1800, D + 100), legend=false)
pa = panel("(a) eigenvalues λ2 … λ5", "|λ − λ(exact)|")
pb = panel("(b) right eigenvectors", "1 − |⟨r|r(exact)⟩|")
pc = panel("(c) left eigenvectors", "1 − |⟨l|l(exact)⟩|")
for (k, m) in enumerate(METHODS)
    rs = results[m.key]
    curve!(pa, rs, r -> r.λerr; color=k, label=m.label * ", SCI")
    curve!(pa, rs, r -> r.λerr_low; color=k, label=m.label * ", SCI + Löwdin", ls=:dash, marker=:diamond, ribbon=false)
    curve!(pb, rs, r -> r.r_inf; color=k, label=m.label)
    curve!(pc, rs, r -> r.l_inf; color=k, label=m.label)
end
for p in (pa, pb, pc)
    vline!(p, [D], color=:gray50, ls=:dot, lw=1, label="")
end
plot!(pa, legend=:bottomleft, legendfontsize=7)
fig = plot(pa, pb, pc, layout=(1, 3), size=(1250, 400), bottom_margin=6mm, left_margin=6mm)
savefig_both(fig, "selection_comparison_L$(Lc)_N$(Nb)")
