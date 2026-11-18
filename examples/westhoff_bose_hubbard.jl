# Recreation of Figs. 2(b) and 4 of Westhoff, Moroder, Schollwöck, Paeckel,
# "Tensor-Network Framework for Lindbladian Spectra and Steady States", PRX 16, 041020 (2026),
# with selected CI (+ Löwdin) in place of CLIK-MPS.
#
# Model: dissipative Bose-Hubbard chain, Eqs. (12)-(13), L = 5 sites, N = 4 bosons,
# J = U = κ = 1 (70 Fock states, Liouville dimension 4900).
#
# Baseline, as in the paper: Krylov space spanned by real-time-evolved states ρ(kδt),
# δt = 0.1, T = 4 (α = 0), from 50 random initial states; left eigenvectors from the same
# construction with L†. Here the evolution is exact (no MPS truncation).
#
# SCI works in the momentum (standing-wave) Fock basis, where the slow modes are compact
# (in the site basis 99% of each mode's weight needs ~75% of all dyads). Seeds: all
# populations |m⟩⟨m| (kept, so L_PP conserves trace) plus coherences with the condensate
# |N,0,…⟩, which reach the reflection-odd sector holding λ₂ (populations alone are even).
#
#   julia --project=examples examples/westhoff_bose_hubbard.jl
include("common.jl")
using SparseArrays, Random, Serialization
import KrylovKit
using OpenSCI: biorthogonal_eigen

const Lc, Nb = 5, 4
const NJ = 5                                  # modes compared: λ₁ … λ₅
const NRAND = 50
const T, δt = 4.0, 0.1
const SCI_EPS = [1e-2, 3e-3, 1e-3]            # selection thresholds ϵ_add for the SCI bars
const R = 10                                  # SCI states kept (λ₁…λ₅ up to conjugation need 8)

Λ, states = dissipative_bose_hubbard(Lc, Nb; J=1, U=1, κ=1, basis=:momentum)
d = length(states)
A = sparse(Λ)
D = size(A, 1)
@printf("L = %d, N = %d: %d Fock states, Liouville dimension %d\n", Lc, Nb, d, D)

# the paper lists eigenvalues by decreasing real part, upper half plane only
# (real eigenvalues from Arnoldi carry ~1e-8 imaginary noise)
upper_idx(λ) = filter(i -> imag(λ[i]) > -1e-6, sortperm(λ, by=z -> (-real(z), -imag(z))))
infid(a, b) = 1 - abs(dot(a, b)) / (norm(a) * norm(b))
cached(f, name) = (p = joinpath(FIGDIR, name * ".jls"); isfile(p) ? deserialize(p) : (x = f(); serialize(p, x); x))

# ---------------------------------------------------------------- exact reference
ex = cached("bh_exact") do
    @time λ, V, W = biorthogonal_eigen(A, 16; dense_max=0)       # Arnoldi, converged to 1e-12
    (λ=λ, V=V, W=W)
end
jex = upper_idx(ex.λ)[1:NJ]
λex = ex.λ[jex]
println("exact λ₁…λ₅: ", round.(λex, digits=4))

# ---------------------------------------------------------------- real-time Krylov baseline
function krylov_ritz(B, x0)
    xs = [x0]
    x = x0
    for _ in 1:round(Int, T / δt)
        x, _ = KrylovKit.exponentiate(B, δt, x; tol=1e-13)
        push!(xs, x)
    end
    F = svd(reduce(hcat, xs))
    Q = F.U[:, F.S .> 1e-12 * F.S[1]]
    E = eigen(Q' * (B * Q))
    return E.values, Q * E.vectors
end

base = cached("bh_krylov_baseline") do
    rng = MersenneTwister(2026)
    Aadj = sparse(A')
    runs = map(1:NRAND) do r
        G = randn(rng, ComplexF64, d, d)
        ρ0 = vec(G * G' / tr(G * G'))                     # random density matrix
        X = randn(rng, ComplexF64, d, d); X = vec(X + X')  # random Hermitian (for L†)
        λr, Vr = krylov_ritz(A, ρ0)
        μl, Ul = krylov_ritz(Aadj, X)                      # L† u = conj(λ) u  ⇒  u† L = λ u†
        ir = upper_idx(λr)[1:NJ]
        il = upper_idx(conj.(μl))[1:NJ]
        (λ_all=λr, λ=λr[ir],
         r_inf=[infid(Vr[:, ir[j]], ex.V[:, jex[j]]) for j in 1:NJ],
         l_inf=[infid(conj.(Ul[:, il[j]]), ex.W[jex[j], :]) for j in 1:NJ])
    end
    runs
end
geo(xs) = exp(mean(log.(max.(xs, 1e-16))))          # average of the logarithm, as in the paper
mean(x) = sum(x) / length(x)
base_λerr = [geo([abs(r.λ[j] - λex[j]) for r in base]) for j in 1:NJ]
base_rinf = [geo([r.r_inf[j] for r in base]) for j in 1:NJ]
base_linf = [geo([r.l_inf[j] for r in base]) for j in 1:NJ]

# ---------------------------------------------------------------- selected CI (+ Löwdin)
c = findfirst(==([Nb; zeros(Int, Lc - 1)]), states)      # condensate |N,0,…,0⟩ in momentum modes
seeds = unique([[(i, i) for i in 1:d]; [(c, m) for m in 1:d]; [(m, c) for m in 1:d]])
embed(res, v) = (x = zeros(ComplexF64, D); for (k, (a, b)) in enumerate(res.basis); x[a+(b-1)*d] = v[k]; end; x)

sci = cached("bh_sci") do
    map(SCI_EPS) do ϵ
        hist = Tuple{Int,Vector{ComplexF64}}[]
        t = @elapsed res = selected_ci(Λ, seeds, R; ϵ_add=ϵ, ϵ_clip=ϵ / 10, max_iter=20, verbose=false,
                                       callback=r -> push!(hist, (length(r), copy(r.λ))))
        lw = lowdin_correction(Λ, res)
        js = upper_idx(res.λ)[1:NJ]                  # Löwdin corrects state by state: same indices
        @printf(" SCI ϵ = %.0e: dim P = %d (%d iterations%s, %.1f s)\n", ϵ, length(res), res.iteration,
                res.converged ? ", converged" : "", t)
        (ϵ=ϵ, dimP=length(res), λ=res.λ, λlow=lw.λ, hist=hist, converged=res.converged,
         λerr=[abs(res.λ[js[j]] - λex[j]) for j in 1:NJ],
         λerr_low=[abs(lw.λ[js[j]] - λex[j]) for j in 1:NJ],
         r_inf=[infid(embed(res, res.V[:, js[j]]), ex.V[:, jex[j]]) for j in 1:NJ],
         l_inf=[infid(embed(res, res.W[js[j], :]), ex.W[jex[j], :]) for j in 1:NJ])
    end
end

println("\n deviation |λ_j − λ_exact|:")
@printf(" %-28s", ""); for j in 1:NJ; @printf("   λ%d     ", j); end; println()
row(name, v) = (@printf(" %-28s", name); foreach(x -> @printf(" %9.2e", x), v); println())
row("real-time Krylov (geo mean)", base_λerr)
for s in sci
    row(@sprintf("SCI ϵ=%.0e (dim P %d)", s.ϵ, s.dimP), s.λerr)
    row(@sprintf("SCI+Löwdin ϵ=%.0e", s.ϵ), s.λerr_low)
end

# ---------------------------------------------------------------- Fig. 2(b)
best = sci[1]                     # coarsest threshold: the most compressed P
win = (x=(-2.3, 0.15), y=(-0.95, 2.25))
inwin(z) = win.x[1] ≤ real(z) ≤ win.x[2] && win.y[1] ≤ imag(z) ≤ win.y[2]
cloud = filter(inwin, reduce(vcat, [r.λ_all for r in base]))
λall_ex = filter(inwin, ex.λ)

p = scatter(real(cloud), imag(cloud), ms=4, color=:steelblue, alpha=0.08, msw=0,
            label="α = 0, T = 4, random initial", xlims=win.x, ylims=win.y,
            xlabel="Re(λ)", ylabel="Im(λ)", size=(560, 470), legend=:bottomleft,
            title="Dissipative Bose-Hubbard, L = $Lc, N = $Nb")
vline!(p, [0], color=:gray50, ls=:dash, lw=1, label="")
hline!(p, [0], color=:gray50, ls=:dash, lw=1, label="")
scatter!(p, real(λall_ex), imag(λall_ex), ms=6, color=3, msw=0, label="exact")
scatter!(p, real(best.λ), imag(best.λ), marker=:xcross, ms=8, msw=3, color=4,
         label="SCI, dim P = $(best.dimP) of $D")
scatter!(p, real(best.λlow), imag(best.λlow), marker=:diamond, ms=4, msw=0.5, color=:orange,
         label="SCI + Löwdin")

# inset: −Re λ_j over SCI iterations (paper: over maximal time T, for L = 20)
dims = [h[1] for h in best.hist]
conv = [[-real(h[2][upper_idx(h[2])[min(j, end)]]) for h in best.hist] for j in 2:NJ]
plot!(p, inset=(1, bbox(0.50, 0.05, 0.46, 0.34)), subplot=2)
for (k, cv) in enumerate(conv)
    plot!(p[2], 1:length(cv), cv, marker=:circle, ms=2.5, lw=1.5, color=k, label="λ$(k+1)")
    hline!(p[2], [-real(λex[k+1])], color=k, ls=:dot, lw=1, label="")
end
plot!(p[2], xlabel="SCI iteration", ylabel="−Re λ", legend=false, guidefontsize=7, tickfontsize=6,
      framestyle=:box)
savefig_both(p, "westhoff_fig2b_L$(Lc)_N$(Nb)")

# ---------------------------------------------------------------- Fig. 4
# Bars below the panel floor are drawn at the floor and noted (SCI keeps every population
# in P, so trace is conserved and λ₁ and its left eigenvector, the identity, are exact).
function bars(panel_data, labels, colors; ylabel, title, floorv, note="")
    n = length(panel_data)
    w = 0.8 / n
    q = plot(xticks=(1:NJ, ["λ$j" for j in 1:NJ]), yscale=:log10, ylabel=ylabel, title=title,
             legend=false, ylims=(floorv, 3), yminorgrid=false)
    for (k, (v, c)) in enumerate(zip(panel_data, colors))
        x = (1:NJ) .- 0.4 .+ (k - 0.5) * w
        bar!(q, x, max.(v, floorv), bar_width=w, fillto=floorv, color=c, lw=0, label=labels[k])
    end
    isempty(note) || annotate!(q, [(0.62 + w, floorv * 4, text(note, 7, :left, :gray30, "Computer Modern"))])
    return q
end

cols_sci = [:mediumpurple1, :purple3, :indigo]
cols_low = [:navajowhite2, :darkorange, :orangered3]
lab_sci = [@sprintf("SCI, dim P = %d", s.dimP) for s in sci]
lab_low = [@sprintf("SCI + Löwdin, dim P = %d", s.dimP) for s in sci]
pa = bars([[base_λerr]; [s.λerr for s in sci]; [s.λerr_low for s in sci]],
          ["real time, T = 4, random initial"; lab_sci; lab_low],
          [:steelblue; cols_sci; cols_low], floorv=1e-7, note="SCI: < 1e-12",
          ylabel="deviation from exact", title="(a) eigenvalues   (Liouville dimension $D)")
plot!(pa, legend=:outerright, legendfontsize=7)
pb = bars([[base_rinf]; [s.r_inf for s in sci]], ["baseline"; lab_sci], [:steelblue; cols_sci],
          floorv=1e-12, ylabel="1 − |⟨r(approx)|r(exact)⟩|", title="(b) right eigenvectors")
pc = bars([[base_linf]; [s.l_inf for s in sci]], ["baseline"; lab_sci], [:steelblue; cols_sci],
          floorv=1e-10, note="SCI: exact", ylabel="1 − |⟨l(approx)|l(exact)⟩|", title="(c) left eigenvectors")
fig4 = plot(pa, plot(pb, pc, layout=(1, 2)), layout=grid(2, 1, heights=[0.5, 0.5]), size=(980, 760))
savefig_both(fig4, "westhoff_fig4_L$(Lc)_N$(Nb)")
