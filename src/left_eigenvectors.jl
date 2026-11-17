using LinearAlgebra
using Arpack
using OrderedCollections

"""
    left_eigenvectors(L::Lindbladian{N}, v; dense_max=3000) -> w

Biorthogonal left eigenvectors of the subspace-projected Lindbladian `L_PP = P L P`
that pair with the right eigenvectors stored in `v` (e.g. as returned by
[`selected_ci`](@ref)). `v` may be a `SparseDyadVectors` or a `SparsePauliVectors`.

`L_PP` is not normal, so its left eigenvectors are *not* the pseudoinverse of the `R`
retained right eigenvectors: `pinv(V)` only enforces `W·V = I` on the kept block (it is
the least-squares / orthogonal projection onto span(V)), whereas a true left eigenvector
`wᵢ L_PP = λᵢ wᵢ` must also annihilate every *discarded* right eigenvector.

Algorithm (no inverse of the full eigenvector matrix is formed):
1. Solve the left eigenproblem directly, `transpose(L_PP) uᵢ = λᵢ uᵢ`  ⇒  `wᵢ = uᵢᵀ`
   (dense `eigen` for `dim ≤ dense_max`, Arpack `eigs` with `nev = R` otherwise).
2. Pick the `R` left eigenvectors whose eigenvalues match the Rayleigh quotients of the
   columns of `V`.
3. Biorthonormalize with an `R×R` solve, `W = (W_sel V) \\ W_sel`, so that `W·V = I`.
   This matrix is invertible whenever `L_PP` is non-defective (wᵢ·vᵢ ≠ 0), and the solve
   also mixes rows correctly inside degenerate eigenspaces.

The result uses the same layout as `v` — `w[key][i]` is component `key` of the `i`-th
left eigenvector — and is contracted *without* complex conjugation:
`⟨⟨wᵢ|x⟩⟩ = Σ_k w[k][i] * x[k]`.
"""
function left_eigenvectors(L::Lindbladian{N}, v::Union{SparseDyadVectors{N,T}, SparsePauliVectors{N,T}};
                           dense_max=3000) where {N,T}
    V = Matrix(v)
    dim, R = size(V)
    Lpp = build_subspace_L(L, v)

    # 1. left eigenpairs:  transpose(L_PP) u = λ u   ⇔   uᵀ L_PP = λ uᵀ
    if dim <= dense_max
        λl, U = eigen(Matrix(transpose(Lpp)))
    else
        λl, U = eigs(Matrix(transpose(Lpp)), nev=R, which=:LR, maxiter=3000, tol=1e-10)
    end

    # 2. match each kept right vector to a left eigenvector by eigenvalue
    λv = [dot(V[:, j], Lpp * V[:, j]) / dot(V[:, j], V[:, j]) for j in 1:R]
    idx = Int[]
    for j in 1:R
        order = sortperm(abs.(λl .- λv[j]))
        push!(idx, first(Iterators.filter(i -> !(i in idx), order)))
    end
    Wsel = Matrix(transpose(U[:, idx]))          # R × dim, rows wᵢ

    # 3. biorthonormalize against the stored right vectors: W·V = I  (R×R solve)
    W = (Wsel * V) \ Wsel

    w = empty(v)
    for (i, k) in enumerate(keys(v))
        w[k] = W[:, i]
    end
    return w
end
