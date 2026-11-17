"""
    SCIResult{K}

Outcome of [`selected_ci`](@ref) (or of one of its iterations).

- `basis::Vector{K}`: the selected P space (`DyadBasis` or `PauliBasis` keys)
- `λ`: the `R` eigenvalues of `L_PP` with largest real part, sorted by decreasing real part
  (so `λ[1]` is the steady state, `λ[2]` the slowest decaying mode, ...)
- `V`: right eigenvectors (`length(basis) × R`)
- `W`: biorthogonal left eigenvectors (`R × length(basis)`), `W*V = I`
- `iteration`, `converged`
"""
struct SCIResult{K}
    basis::Vector{K}
    pos::Dict{K,Int}
    λ::Vector{ComplexF64}
    V::Matrix{ComplexF64}
    W::Matrix{ComplexF64}
    iteration::Int
    converged::Bool
end

Base.length(r::SCIResult) = length(r.basis)
nstates(r::SCIResult) = length(r.λ)

function Base.show(io::IO, ::MIME"text/plain", r::SCIResult{K}) where K
    @printf(io, "SCIResult{%s}: dim P = %d, %d states, iteration %d%s\n", nameof(K), length(r),
            nstates(r), r.iteration, r.converged ? " (converged)" : "")
    for (i, λ) in enumerate(r.λ)
        @printf(io, "  λ[%d] = % .10f %+.10fi\n", i, real(λ), imag(λ))
    end
end

"""
    selected_ci(L, init, R; basis=DyadBasis, ϵ_add=1e-2, ϵ_clip=1e-4, max_iter=10,
                selection=:amplitude, dense_max=3000, callback=nothing, verbose=true)

Selected-CI for the `R` slowest eigenmodes of the Lindbladian `L`.

Start from the keys of `init` (any PauliOperators operator, e.g. `Dyad(N,0,0)`, expanded in
`basis`). Each iteration:
1. **clip**: drop keys whose largest eigenvector amplitude is below `ϵ_clip`;
2. **expand**: add every external key `q` whose amplitude exceeds `ϵ_add`;
   the amplitude is `|(L V)_q|` for `selection=:amplitude`, or the first-order
   perturbative estimate `|(L V)_q / (λ − L_qq)|` for `selection=:perturbative`;
3. **diagonalize** `L_PP`, keeping the `R` eigenvalues with largest real part and their
   biorthogonal left/right eigenvectors.

Stops when the space no longer changes. `callback(result::SCIResult)` is called after every
iteration (e.g. to record convergence or apply a Löwdin correction per iteration).

Defaults reproduce the old `selected_ci(…; ϵdiscard=1e-4)`.
"""
function selected_ci(Λ::Lindbladian{N}, init, R::Integer; basis::Type=DyadBasis,
                     ϵ_add=1e-2, ϵ_clip=1e-4, max_iter=10, selection::Symbol=:amplitude,
                     dense_max=3000, callback=nothing, verbose=true) where N
    selection in (:amplitude, :perturbative) || throw(ArgumentError("selection must be :amplitude or :perturbative"))
    K = basis{N}
    t = SuperOpTerms(Λ)

    c0 = coefficients(basis, init)
    P = collect(keys(c0))::Vector{K}
    V = reshape(normalize([c0[k] for k in P]), :, 1)
    λ = ComplexF64[]
    res = nothing
    verbose && @printf(" %5s %8s %22s %10s\n", "iter", "dim P", "λ[min(2,R)]", "dim Q")

    for it in 1:max_iter
        # 1. clip
        keep = [maximum(abs, @view V[i, :]) > ϵ_clip for i in eachindex(P)]
        P, V = P[keep], V[keep, :]
        pos = Dict{K,Int}(k => i for (i, k) in enumerate(P))

        # 2. expand
        ext = external_action(t, P, pos, V)
        newP = copy(P)
        for (q, a) in ext
            if selection === :perturbative && !isempty(λ)
                a = a ./ (λ[1:length(a)] .- diag_element(t, q))
            end
            maximum(abs, a) > ϵ_add && push!(newP, q)
        end

        if res !== nothing && length(newP) == length(res.basis) && Set(newP) == Set(res.basis)
            res = SCIResult{K}(res.basis, res.pos, res.λ, res.V, res.W, res.iteration, true)
            verbose && println(" converged: P space unchanged")
            break
        end

        # 3. diagonalize L_PP
        P = newP
        pos = Dict{K,Int}(k => i for (i, k) in enumerate(P))
        Lpp = subspace_matrix(t, P)
        λ, V, W = biorthogonal_eigen(Lpp, R; dense_max=dense_max)
        res = SCIResult{K}(P, pos, λ, V, W, it, false)
        verbose && @printf(" %5d %8d % .10f%+.10fi %10d\n", it, length(P),
                           real(λ[min(2, end)]), imag(λ[min(2, end)]), length(ext))
        callback === nothing || callback(res)
    end
    return res
end
