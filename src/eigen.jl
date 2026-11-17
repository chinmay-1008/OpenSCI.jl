"""
    biorthogonal_eigen(A, R; dense_max=3000, tol=1e-10) -> (λ, V, W)

The `R` eigenvalues of the (non-normal) matrix `A` with largest real part, sorted by
decreasing real part, with right eigenvectors `V` (n×R, unit columns) and true left
eigenvectors `W` (R×n) normalized so that `W*V = I`.

`W` is *not* `pinv(V)`: a left eigenvector must also annihilate every discarded right
eigenvector, which `pinv` only enforces on the kept block. Here each left vector comes
from the left eigenproblem `transpose(A) u = λ u`; within a (near-)degenerate cluster the
left and right bases are paired with a small `k×k` solve, so no full inverse is formed.

Dense `eigen` is used for `n ≤ dense_max`, KrylovKit's Arnoldi otherwise.
"""
function biorthogonal_eigen(A::AbstractMatrix, R::Integer; dense_max=3000, tol=1e-10)
    n = size(A, 1)
    R = min(R, n)
    if n <= dense_max
        Ad = Matrix(A)
        λr, Vr = eigen(Ad)
        λl, Ul = eigen(Matrix(transpose(Ad)))
    else
        λr, Vr = _krylov_eigs(A, R)
        λl, Ul = _krylov_eigs(sparse(transpose(A)), R)
    end
    order = sortperm(λr, by=z -> (-real(z), -imag(z)))[1:R]
    λ = λr[order]
    V = Vr[:, order]
    W = zeros(eltype(V), R, n)
    ctol = 1e-7 * max(1.0, maximum(abs, λ))
    done = falses(R)
    for j in 1:R
        done[j] && continue
        # whole cluster of (near-)degenerate eigenvalues around λ[j]
        cr = findall(z -> abs(z - λ[j]) < ctol, λr)
        cl = findall(z -> abs(z - λ[j]) < ctol, λl)
        if length(cr) != length(cl)
            @warn "left/right cluster size mismatch at λ = $(λ[j]) ($(length(cl)) vs $(length(cr))); pairing by nearest eigenvalue"
            cl = [argmin(abs.(λl .- λ[j]))]
            cr = [order[j]]
        end
        Ut = transpose(Ul[:, cl])                # k × n
        Wc = (Ut * Vr[:, cr]) \ Ut               # rows dual to Vr[:, cr]
        for (a, i) in enumerate(cr)
            jj = findfirst(==(i), order)
            jj === nothing && continue           # cluster member outside the kept R
            W[jj, :] = Wc[a, :]
            done[jj] = true
        end
    end
    return λ, V, W
end

function _krylov_eigs(A, R)
    n = size(A, 1)
    x0 = normalize!(randn(ComplexF64, n))
    howmany = min(R + 4, n)
    vals, vecs, info = KrylovKit.eigsolve(A, x0, howmany, :LR;
                                          krylovdim=max(30, 2howmany + 10), tol=1e-12, maxiter=500)
    info.converged < R && @warn "Arnoldi: only $(info.converged) of $R eigenpairs converged"
    return vals, reduce(hcat, vecs)
end
