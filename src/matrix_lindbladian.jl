"""
    MatrixLindbladian(H, L, γ)

Lindbladian given by sparse matrices over a user-chosen set of `d` basis states (e.g. the
fixed-particle-number Fock states of a bosonic chain), for models where a Pauli-string
encoding blows up (a binary-encoded boson jump operator is hundreds of Pauli strings).

SCI keys are `(ket, bra)` index pairs, `|ket⟩⟨bra|`; seeds and observables are plain
`d × d` matrices (or a single `(ket, bra)` pair as a seed). The SCI, Löwdin and dynamics
code is shared with the Pauli backend.
"""
struct MatrixLindbladian <: AbstractLindbladian
    H::SparseMatrixCSC{ComplexF64,Int}
    L::Vector{SparseMatrixCSC{ComplexF64,Int}}
    γ::Vector{Float64}
    function MatrixLindbladian(H, L, γ)
        length(L) == length(γ) || throw(ArgumentError("need one rate per jump operator"))
        d = size(H, 1)
        all(size(Lk) == (d, d) for Lk in L) || throw(DimensionMismatch("jump operators must be $d × $d"))
        new(sparse(ComplexF64.(H)), [sparse(ComplexF64.(Lk)) for Lk in L], Float64.(γ))
    end
end

const MatKey = Tuple{Int,Int}

"""
    change_basis(Λ, U) -> MatrixLindbladian

The same Lindbladian in the basis `|m⟩ = Σ_s U[m,s]^* |s⟩` (`H → U H U†`, `Lₖ → U Lₖ U†`);
`U` must be unitary. The spectrum is unchanged; SCI behaviour can change a lot.
"""
function change_basis(Λ::MatrixLindbladian, U::AbstractMatrix)
    rot(X) = droptol!(sparse(U * X * U'), 1e-13)
    return MatrixLindbladian(rot(Λ.H), [rot(Lk) for Lk in Λ.L], Λ.γ)
end

hilbert_dim(Λ::MatrixLindbladian) = size(Λ.H, 1)

function Base.show(io::IO, ::MIME"text/plain", Λ::MatrixLindbladian)
    @printf(io, "MatrixLindbladian: d = %d states, nnz(H) = %d, %d jump operators\n",
            hilbert_dim(Λ), nnz(Λ.H), length(Λ.L))
end

# L(|a⟩⟨b|) = -i Heff|a⟩⟨b| + i|a⟩⟨b|Heff† + Σₖ γₖ Lₖ|a⟩⟨b|Lₖ†,   Heff = H - (i/2) Σₖ γₖ Lₖ†Lₖ
struct MatrixTerms <: AbstractTerms
    Heff::SparseMatrixCSC{ComplexF64,Int}
    L::Vector{SparseMatrixCSC{ComplexF64,Int}}
    γ::Vector{Float64}
end

function _terms(Λ::MatrixLindbladian)
    G = isempty(Λ.L) ? spzeros(ComplexF64, size(Λ.H)...) : sum(γ / 2 * (Lk' * Lk) for (Lk, γ) in zip(Λ.L, Λ.γ))
    return MatrixTerms(droptol!(Λ.H - im * G, 1e-15), Λ.L, Λ.γ)
end
# L†(X) = i Heff† X − i X Heff + Σₖ γₖ Lₖ† X Lₖ: the same form with Heff → −Heff†, Lₖ → Lₖ†
_adjoint(t::MatrixTerms) = MatrixTerms(sparse(-t.Heff'), [sparse(Lk') for Lk in t.L], t.γ)

_keytype(::MatrixLindbladian, ::Type) = MatKey
_all_keys(Λ::MatrixLindbladian, ::Type) = [(i, j) for j in 1:hilbert_dim(Λ) for i in 1:hilbert_dim(Λ)]
_basis_type(::Type{MatKey}) = MatKey

@inline function foreach_image(f::F, t::MatrixTerms, key::MatKey) where F
    a, b = key
    H, rv, nz = t.Heff, rowvals(t.Heff), nonzeros(t.Heff)
    @inbounds begin
        for i in nzrange(H, a)                      # -i Heff |a⟩⟨b|
            f((rv[i], b), -im * nz[i])
        end
        for i in nzrange(H, b)                      # +i |a⟩⟨b| Heff† = i Σ_c conj(Heff[c,b]) |a⟩⟨c|
            f((a, rv[i]), im * conj(nz[i]))
        end
        for (Lk, γ) in zip(t.L, t.γ)                # γ Lₖ|a⟩⟨b|Lₖ†
            rk, vk = rowvals(Lk), nonzeros(Lk)
            for i in nzrange(Lk, a), j in nzrange(Lk, b)
                f((rk[i], rk[j]), γ * vk[i] * conj(vk[j]))
            end
        end
    end
end

"""Dense `d² × d²` matrix in `vec(ρ)` ordering (the `basis` argument is ignored)."""
Base.Matrix(Λ::MatrixLindbladian, ::Type=DyadBasis) = Matrix(sparse(Λ))

# seeds / initial states and observables as plain matrices
coefficients(::Type{MatKey}, k::MatKey) = Dict{MatKey,ComplexF64}(k => 1.0)
function coefficients(::Type{MatKey}, ρ::AbstractMatrix)
    out = Dict{MatKey,ComplexF64}()
    for j in axes(ρ, 2), i in axes(ρ, 1)
        abs(ρ[i, j]) > 1e-15 && (out[(i, j)] = ρ[i, j])
    end
    return out
end
_trace_with(O::AbstractMatrix, key::MatKey) = O[key[2], key[1]]      # tr(O |a⟩⟨b|) = ⟨b|O|a⟩

# Operator-level action (dense), handy for checks
function Base.:*(Λ::MatrixLindbladian, ρ::AbstractMatrix)
    dρ = -im * (Λ.H * ρ - ρ * Λ.H)
    for (Lk, γ) in zip(Λ.L, Λ.γ)
        LL = Lk' * Lk
        dρ += γ * (Lk * ρ * Lk' - (LL * ρ + ρ * LL) / 2)
    end
    return dρ
end
