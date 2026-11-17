# Operators <-> coefficient vectors in a chosen basis, and subspace matrices of L.
#
# Coefficients are defined by  O = Σ_key c_key · key,  for both bases. Because all Pauli
# strings have the same Hilbert-Schmidt norm (2^N), the matrix of L is the same whether one
# uses Pauli strings or normalized Pauli strings, so SCI thresholds mean the same thing.

_basis_type(::Type{<:DyadBasis}) = DyadBasis
_basis_type(::Type{<:PauliBasis}) = PauliBasis

"""
    coefficients(basis, O) -> Dict{key, ComplexF64}

Expand a PauliOperators object (`Pauli`, `PauliBasis`, `PauliSum`, `Dyad`, `DyadBasis`,
`DyadSum`) in `basis` (`DyadBasis` or `PauliBasis`).
"""
coefficients(basis::Type, O::Union{Pauli,PauliBasis}) = coefficients(basis, PauliSum(O))
coefficients(basis::Type, O::Union{Dyad,DyadBasis}) = coefficients(basis, DyadSum(O))

function coefficients(::Type{PauliBasis}, O::PauliSum{N}) where N
    return Dict{PauliBasis{N},ComplexF64}(k => c for (k, c) in O)
end

function coefficients(::Type{DyadBasis}, O::DyadSum{N}) where N
    return Dict{DyadBasis{N},ComplexF64}(k => c for (k, c) in O)
end

# P = Σ_k P|k⟩⟨k|
function coefficients(::Type{DyadBasis}, O::PauliSum{N}) where N
    out = Dict{DyadBasis{N},ComplexF64}()
    for (P, c) in O, k in 0:2^N-1
        d = P * Dyad(N, k, k)
        out[DyadBasis(d)] = get(out, DyadBasis(d), 0.0im) + c * coeff(d)
    end
    filter!(kv -> abs(kv.second) > 1e-15, out)
end

# |k⟩⟨b| = 2^-N Σ_P tr(P |k⟩⟨b|) P ; only P with x-bits = k ⊻ b contribute
function coefficients(::Type{PauliBasis}, O::DyadSum{N}) where N
    out = Dict{PauliBasis{N},ComplexF64}()
    for (d, c) in O
        x = d.ket.v ⊻ d.bra.v
        for z in 0:2^N-1
            P = PauliBasis{N}(z, x)
            a = tr(P * Dyad(d)) / 2^N
            out[P] = get(out, P, 0.0im) + c * a
        end
    end
    filter!(kv -> abs(kv.second) > 1e-15, out)
end

"""tr(O · key) for a PauliOperators object `O` and a single basis key."""
_trace_with(O::Union{PauliSum,DyadSum}, key) = sum((c * tr(k * _elem(key)) for (k, c) in O); init=0.0im)
_trace_with(O, key) = _trace_with(O isa Union{Pauli,PauliBasis} ? PauliSum(O) : DyadSum(O), key)

"""
    subspace_matrix(L, rows, cols=rows) -> SparseMatrixCSC

`⟨⟨rᵢ|L|cⱼ⟩⟩` for basis keys `rows`, `cols` (vectors of `DyadBasis` or `PauliBasis`).
`L` may be a `Lindbladian` or precomputed `SuperOpTerms`.
"""
subspace_matrix(Λ::Lindbladian, rows, cols=rows) = subspace_matrix(SuperOpTerms(Λ), rows, cols)

function subspace_matrix(t::SuperOpTerms, rows::AbstractVector{K}, cols::AbstractVector{K}=rows) where K
    rpos = Dict{K,Int}(k => i for (i, k) in enumerate(rows))
    I, J, V = Int[], Int[], ComplexF64[]
    for (j, b) in enumerate(cols)
        foreach_image(t, b) do k, c
            i = get(rpos, k, 0)
            if i > 0
                push!(I, i); push!(J, j); push!(V, c)
            end
        end
    end
    A = sparse(I, J, V, length(rows), length(cols))
    return droptol!(A, 1e-14)
end

"""
    external_action(t, P, pos, V) -> Dict{key, Vector}

`(L V)_q` for every key `q` reachable from the space `P` but not in it (the Q space),
for each column of `V` (`length(P) × R`).
"""
function external_action(t::SuperOpTerms, P::AbstractVector{K}, pos::Dict{K,Int}, V::AbstractMatrix) where K
    R = size(V, 2)
    out = Dict{K,Vector{ComplexF64}}()
    for (j, b) in enumerate(P)
        vj = @view V[j, :]
        foreach_image(t, b) do k, c
            haskey(pos, k) && return
            acc = get!(() -> zeros(ComplexF64, R), out, k)
            acc .+= c .* vj
        end
    end
    # drop keys whose couplings cancel exactly (e.g. commuting Hamiltonian terms in the Pauli basis)
    return filter!(kv -> maximum(abs, kv.second) > 1e-14, out)
end
