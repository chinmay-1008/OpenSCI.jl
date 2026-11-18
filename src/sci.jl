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
                selection=:amplitude, two_sided=false, keep_seeds=true, dense_max=3000,
                callback=nothing, verbose=true)

Selected-CI for the `R` slowest eigenmodes of the Lindbladian `L`.

Start from the keys of `init` (any PauliOperators operator, e.g. `Dyad(N,0,0)`, expanded in
`basis`; for a [`MatrixLindbladian`](@ref), a density matrix or a `(ket, bra)` index pair),
or from a vector of such seeds, each giving one starting column. Each iteration:
1. **clip**: drop keys whose largest eigenvector amplitude is below `ϵ_clip` (seed keys are
   never dropped when `keep_seeds=true`: seeding with all populations `|i⟩⟨i|` keeps the
   trace functional inside P, so `L_PP` conserves probability and the steady state stays at 0);
2. **expand**: add every external key `q` whose score exceeds `ϵ_add`;
3. **diagonalize** `L_PP`, keeping the `R` eigenvalues with largest real part and their
   biorthogonal left/right eigenvectors.

Scores of an external key `q`, with `vᵢ` the right eigenvectors (unit norm) and `wᵢ` the left
ones (`wᵢ vᵢ = 1`), maximized over the states `i`:
- `selection=:amplitude`: `|(L vᵢ)_q|`
- `selection=:perturbative`: the first-order amplitude `|(L vᵢ)_q / (λᵢ − L_qq)|`
- `selection=:pt2`: the second-order eigenvalue shift that `q` would bring,
  `|(wᵢ L)_q (L vᵢ)_q / (λᵢ − L_qq)|` (the term `q` contributes to [`lowdin_correction`](@ref));
  it has units of a rate, so `ϵ_add` is typically much smaller (≈ 1e-6).

Right eigenvectors alone decide which keys enter P for the first two, but in a non-normal
`L` the left eigenvectors can live elsewhere. `two_sided=true` also scores the left
eigenvectors, normalized to unit norm (`|(ŵᵢ L)_q|`, or divided by `λᵢ − L_qq`), and clips a
key only when it is small in both. `:pt2` uses both sides by construction and clips like
`two_sided`.

Stops when the space no longer changes. `callback(result::SCIResult)` is called after every
iteration (e.g. to record convergence or apply a Löwdin correction per iteration).

Defaults reproduce the old `selected_ci(…; ϵdiscard=1e-4)`.
"""
function selected_ci(Λ::AbstractLindbladian, init, R::Integer; basis::Type=DyadBasis,
                     ϵ_add=1e-2, ϵ_clip=1e-4, max_iter=10, selection::Symbol=:amplitude,
                     two_sided::Bool=false, keep_seeds=true, dense_max=3000, callback=nothing,
                     verbose=true)
    selection in (:amplitude, :perturbative, :pt2) ||
        throw(ArgumentError("selection must be :amplitude, :perturbative or :pt2"))
    left = two_sided || selection === :pt2
    K = _keytype(Λ, basis)
    t = _terms(Λ)

    seeds = [coefficients(_basis_type(K), x) for x in (init isa AbstractVector ? init : [init])]
    P = unique(K[k for c in seeds for k in keys(c)])
    V = reduce(hcat, [normalize([get(c, k, 0.0im) for k in P]) for c in seeds])
    W = zeros(ComplexF64, 0, length(P))      # no left eigenvectors before the first diagonalization
    protected = keep_seeds ? Set(P) : Set{K}()
    λ = ComplexF64[]
    res = nothing
    verbose && @printf(" %5s %8s %22s %10s\n", "iter", "dim P", "λ[min(2,R)]", "dim Q")

    for it in 1:max_iter
        # 1. clip (seed keys are kept when keep_seeds); two-sided: small in V and in Ŵ
        Ŵ = W ./ norm.(eachrow(W))
        keep = map(eachindex(P)) do i
            P[i] in protected && return true
            a = maximum(abs, @view V[i, :])
            left && !isempty(W) && (a = max(a, maximum(abs, @view Ŵ[:, i])))
            return a > ϵ_clip
        end
        P, V, W, Ŵ = P[keep], V[keep, :], W[:, keep], Ŵ[:, keep]
        pos = Dict{K,Int}(k => i for (i, k) in enumerate(P))

        # 2. expand
        ext = external_action(t, P, pos, V)
        extL = left && !isempty(W) ?
               external_left_action(t, P, pos, selection === :pt2 ? W : Ŵ) : empty(ext)
        nR = size(V, 2)
        zR = zeros(ComplexF64, nR)
        function score(q, aR, aL)
            (selection === :amplitude || isempty(λ)) && return max(maximum(abs, aR), maximum(abs, aL))
            d = λ[1:nR] .- diag_element(t, q)
            selection === :pt2 && return maximum(abs, aL .* aR ./ d)
            return max(maximum(abs, aR ./ d), maximum(abs, aL ./ d))
        end
        newP = copy(P)
        for (q, aR) in ext
            score(q, aR, get(extL, q, zR)) > ϵ_add && push!(newP, q)
        end
        nQ = length(ext)
        for (q, aL) in extL                  # keys only the left eigenvectors reach
            haskey(ext, q) && continue
            nQ += 1
            score(q, zR, aL) > ϵ_add && push!(newP, q)
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
                           real(λ[min(2, end)]), imag(λ[min(2, end)]), nQ)
        callback === nothing || callback(res)
    end
    return res
end
