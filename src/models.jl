"""
    heisenberg_1D(N, Jx, Jy, Jz; periodic=true)

`H = -2 Σᵢ (Jx XᵢXᵢ₊₁ + Jy YᵢYᵢ₊₁ + Jz ZᵢZᵢ₊₁)` on a ring (or open chain).
"""
function heisenberg_1D(N, Jx, Jy, Jz; periodic=true)
    H = PauliSum(N)
    for i in 1:(periodic ? N : N - 1)
        j = i % N + 1
        for (J, P) in ((Jx, Pauli(N, X=[i, j])), (Jy, Pauli(N, Y=[i, j])), (Jz, Pauli(N, Z=[i, j])))
            k = PauliBasis(P)
            H[k] = get(H, k, 0.0im) - 2J * coeff(P)
        end
    end
    return H
end

"""
    fock_basis(L, N) -> Vector{Vector{Int}}

All occupation vectors of `N` bosons on `L` sites (lexicographic, descending).
"""
function fock_basis(L, N)
    L == 1 && return [[N]]
    return [vcat(n, rest) for n in N:-1:0 for rest in fock_basis(L - 1, N - n)]
end

"""
    hopping(states, i, j) -> SparseMatrixCSC

`b†ᵢ bⱼ` in the fixed-particle-number basis `states` (`nᵢ` when `i == j`).
"""
function hopping(states, i, j)
    idx = Dict(s => k for (k, s) in enumerate(states))
    I, J, V = Int[], Int[], Float64[]
    for (k, s) in enumerate(states)
        if i == j
            s[i] > 0 && (push!(I, k); push!(J, k); push!(V, s[i]))
        elseif s[j] > 0
            t = copy(s); t[j] -= 1; t[i] += 1
            push!(I, idx[t]); push!(J, k); push!(V, sqrt(s[j] * t[i]))
        end
    end
    return sparse(I, J, V, length(states), length(states))
end

"""
    dissipative_bose_hubbard(L, N; J=1, U=1, κ=1) -> (Λ::MatrixLindbladian, states)

Driven bosons of Westhoff et al., PRX 16, 041020 (2026), Eqs. (12)-(13), on an open chain:

    H  = -J Σⱼ (b†ⱼ₊₁bⱼ + h.c.) + U/2 Σⱼ b†ⱼ²bⱼ²
    Lⱼ = √κ (b†ⱼ₊₁ + b†ⱼ)(bⱼ₊₁ − bⱼ),   j = 1 … L-1

Both conserve the particle number, so everything lives in the `N`-boson sector `states`.

`basis = :site` uses Fock states of the lattice sites; `basis = :momentum` uses Fock states of
the open-chain standing waves `φₖ(j) = √(2/(L+1)) sin(πkj/(L+1))` (exact many-body rotation,
same spectrum). The slow modes are far more compact in the momentum basis (the steady state
is close to a condensate), which is what makes selected CI effective for this model.
"""
function dissipative_bose_hubbard(L, N; J=1.0, U=1.0, κ=1.0, basis::Symbol=:site)
    basis in (:site, :momentum) || throw(ArgumentError("basis must be :site or :momentum"))
    states = fock_basis(L, N)
    h(i, j) = hopping(states, i, j)
    H = -J * sum(h(j + 1, j) + h(j, j + 1) for j in 1:L-1) +
        U / 2 * sum(h(j, j) * (h(j, j) - I) for j in 1:L)
    jumps = [h(j + 1, j + 1) - h(j + 1, j) + h(j, j + 1) - h(j, j) for j in 1:L-1]
    Λ = MatrixLindbladian(H, jumps, fill(κ, L - 1))
    basis === :site && return Λ, states
    return change_basis(Λ, fock_rotation(states, standing_waves(L))), states
end

"""Single-particle eigenmodes of open-chain hopping: `φ[j, k] = √(2/(L+1)) sin(πkj/(L+1))`."""
standing_waves(L) = [sqrt(2 / (L + 1)) * sin(π * k * j / (L + 1)) for j in 1:L, k in 1:L]

"""
    fock_rotation(states, φ) -> Matrix

Many-body unitary on the fixed-N Fock space `states` induced by the single-particle change of
basis `b†ⱼ = Σₖ φ[j,k] c†ₖ` (orthogonal `φ`): `U[m, s] = ⟨m (modes k)|s (sites j)⟩`.
"""
function fock_rotation(states, φ)
    L, N = length(states[1]), sum(states[1])
    idx = Dict(s => i for (i, s) in enumerate(states))
    U = zeros(eltype(φ), length(states), length(states))
    for (si, s) in enumerate(states)
        sites = reduce(vcat, [fill(j, s[j]) for j in 1:L])
        norm_s = sqrt(prod(factorial.(s)))
        for ks in Iterators.product(fill(1:L, N)...)        # expand Πᵢ b†_{jᵢ} = Πᵢ Σₖ φ[jᵢ,k] c†ₖ
            amp = prod(φ[sites[i], ks[i]] for i in 1:N)
            amp == 0 && continue
            m = zeros(Int, L)
            foreach(k -> m[k] += 1, ks)
            U[idx[m], si] += amp * sqrt(prod(factorial.(m))) / norm_s
        end
    end
    return U
end
