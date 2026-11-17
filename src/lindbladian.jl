"""
    Lindbladian{N}

    L(ρ) = -i[H, ρ] + Σₖ γₖ (Lₖ ρ Lₖ† − ½{Lₖ†Lₖ, ρ})

- `H::PauliSum`: Hamiltonian
- `L::Vector{PauliSum}`: jump operators
- `γ::Vector{Float64}`: rates
"""
struct Lindbladian{N}
    H::PauliSum{N,ComplexF64}
    L::Vector{PauliSum{N,ComplexF64}}
    γ::Vector{Float64}
end

Lindbladian(N::Integer) = Lindbladian{N}(PauliSum(N), PauliSum{N,ComplexF64}[], Float64[])

function Lindbladian(H::PauliSum{N}, L::AbstractVector, γ::AbstractVector) where N
    length(L) == length(γ) || throw(ArgumentError("need one rate per jump operator"))
    Λ = Lindbladian(N)
    add_hamiltonian!(Λ, H)
    for (Lk, γk) in zip(L, γ)
        push!(Λ.L, _to_paulisum(Lk))
        push!(Λ.γ, γk)
    end
    return Λ
end

_to_paulisum(p::PauliSum{N}) where N = PauliSum{N,ComplexF64}(k => ComplexF64(c) for (k, c) in p)
_to_paulisum(p::Union{Pauli,PauliBasis}) = PauliSum(p)

function add_hamiltonian!(Λ::Lindbladian{N}, H::PauliSum{N}) where N
    for (p, c) in H
        Λ.H[p] = get(Λ.H, p, 0.0im) + c
    end
    return Λ
end

function add_jump!(Λ::Lindbladian{N}, Lk, γ::Real) where N
    push!(Λ.L, _to_paulisum(Lk))
    push!(Λ.γ, γ)
    return Λ
end

_rates(γ::Real, N) = fill(Float64(γ), N)
function _rates(γ::AbstractVector, N)
    length(γ) == N || throw(ArgumentError("length of γ must equal N = $N"))
    return Float64.(γ)
end

"""Single-qubit dephasing `Zᵢ` on every qubit (rate `γ`, scalar or per-qubit)."""
function add_channel_dephasing!(Λ::Lindbladian{N}, γ) where N
    for (i, g) in enumerate(_rates(γ, N))
        add_jump!(Λ, Pauli(N, Z=[i]), g)
    end
    return Λ
end

"""Single-qubit depolarizing: jumps `Xᵢ, Yᵢ, Zᵢ` on every qubit, each with rate `γ`."""
function add_channel_depolarizing!(Λ::Lindbladian{N}, γ) where N
    for (i, g) in enumerate(_rates(γ, N))
        add_jump!(Λ, Pauli(N, X=[i]), g)
        add_jump!(Λ, Pauli(N, Y=[i]), g)
        add_jump!(Λ, Pauli(N, Z=[i]), g)
    end
    return Λ
end

"""
Amplitude damping `σ⁻ = |0⟩⟨1| = (X + iY)/2` on every qubit, so `γ` is the physical decay rate.
(The old code used `X + iY`, i.e. an effective rate of `4γ`.)
"""
function add_channel_amplitude_damping!(Λ::Lindbladian{N}, γ) where N
    for (i, g) in enumerate(_rates(γ, N))
        σm = PauliSum(N)
        σm[PauliBasis(Pauli(N, X=[i]))] = 0.5
        y = Pauli(N, Y=[i])
        σm[PauliBasis(y)] = 0.5im * coeff(y)
        add_jump!(Λ, σm, g)
    end
    return Λ
end

function Base.show(io::IO, ::MIME"text/plain", Λ::Lindbladian{N}) where N
    @printf(io, "Lindbladian{%d}: %d Hamiltonian terms, %d jump operators\n", N, length(Λ.H), length(Λ.L))
    for (p, c) in Λ.H
        @printf(io, "  H  % .6f %+.6fi  %s\n", real(c), imag(c), string(p))
    end
    for (k, (Lk, γ)) in enumerate(zip(Λ.L, Λ.γ))
        @printf(io, "  L%-3d γ = %.4f :", k, γ)
        for (p, c) in Lk
            @printf(io, "  (% .3f %+.3fi) %s", real(c), imag(c), string(p))
        end
        println(io)
    end
end

function Base.rand(::Type{Lindbladian{N}}; nH=2, nL=2) where N
    Λ = Lindbladian(N)
    for _ in 1:nH
        p = PauliBasis(rand(Pauli{N}))
        Λ.H[p] = get(Λ.H, p, 0.0im) + rand()
    end
    for _ in 1:nL
        add_jump!(Λ, PauliSum(rand(Pauli{N})), rand())
    end
    return Λ
end

# ---------------------------------------------------------------------------
#  Superoperator as a list of Pauli sandwiches:  L(ρ) = Σₜ cₜ Aₜ ρ Bₜ
# ---------------------------------------------------------------------------

"""
    SuperOpTerms(Λ::Lindbladian)

`Λ` rewritten as `L(ρ) = Σₜ cₜ Aₜ ρ Bₜ` with Pauli strings `Aₜ, Bₜ`. Identical sandwiches
are merged (e.g. depolarizing `Lₖ†Lₖ = I` collapses to a constant). Acting on one basis key
is then a single Pauli product per term, the same code for dyads and Pauli strings.
"""
struct SuperOpTerms{N}
    c::Vector{ComplexF64}
    A::Vector{PauliBasis{N}}
    B::Vector{PauliBasis{N}}
end

function SuperOpTerms(Λ::Lindbladian{N}; atol=1e-14) where N
    acc = Dict{Tuple{PauliBasis{N},PauliBasis{N}},ComplexF64}()
    add!(A, B, c) = (acc[(A, B)] = get(acc, (A, B), 0.0im) + c)
    Id = PauliBasis{N}(0, 0)
    for (P, h) in Λ.H                       # -i H ρ + i ρ H
        add!(P, Id, -im * h)
        add!(Id, P, im * h)
    end
    for (Lk, γ) in zip(Λ.L, Λ.γ)
        for (Pj, cj) in Lk, (Pk, ck) in Lk
            add!(Pj, Pk, γ * cj * conj(ck))  # γ Lₖ ρ Lₖ†   (Pauli strings are Hermitian)
            PjPk = Pj * Pk                   # Lₖ†Lₖ = Σ cj* ck Pj Pk
            c = -γ / 2 * conj(cj) * ck * coeff(PjPk)
            add!(PauliBasis(PjPk), Id, c)
            add!(Id, PauliBasis(PjPk), c)
        end
    end
    filter!(kv -> abs(kv.second) > atol, acc)
    ks = collect(keys(acc))
    return SuperOpTerms{N}([acc[k] for k in ks], first.(ks), last.(ks))
end

Base.length(t::SuperOpTerms) = length(t.c)

_isid(p::PauliBasis) = p.z == 0 && p.x == 0
_elem(b::PauliBasis) = Pauli(b)
_elem(b::DyadBasis) = Dyad(b)
_key(x::Pauli) = PauliBasis(x)
_key(x::Dyad) = DyadBasis(x)

"""
    foreach_image(f, terms, b)

Call `f(key, value)` for every term of `L(b)` (duplicates not merged), where `b` is a
`DyadBasis` or `PauliBasis` key.
"""
@inline function foreach_image(f::F, t::SuperOpTerms, b) where F
    x0 = _elem(b)
    @inbounds for i in eachindex(t.c)
        A, B = t.A[i], t.B[i]
        x = _isid(A) ? x0 : A * x0
        x = _isid(B) ? x : x * B
        f(_key(x), t.c[i] * coeff(x))
    end
end

"""Diagonal element `⟨⟨b|L|b⟩⟩`."""
function diag_element(t::SuperOpTerms, b)
    s = 0.0im
    foreach_image((k, c) -> (k == b && (s += c)), t, b)
    return s
end

# all basis keys in canonical (vectorization) order
_all_keys(::Type{DyadBasis}, N) = [DyadBasis(N, k, b) for b in 0:2^N-1 for k in 0:2^N-1]
_all_keys(::Type{PauliBasis}, N) = [PauliBasis{N}(z, x) for x in 0:2^N-1 for z in 0:2^N-1]

"""
    Matrix(Λ, basis=DyadBasis)

Dense `4^N × 4^N` matrix of `Λ`. In the dyad basis the ordering matches `vec(ρ)`
(column-major), so `Matrix(Λ) * vec(Matrix(ρ)) == vec(Matrix(Λ*ρ))`.
"""
function Base.Matrix(Λ::Lindbladian{N}, basis::Type=DyadBasis) where N
    keys = _all_keys(basis, N)
    pos = Dict(k => i for (i, k) in enumerate(keys))
    t = SuperOpTerms(Λ)
    M = zeros(ComplexF64, 4^N, 4^N)
    for (j, b) in enumerate(keys)
        foreach_image((k, c) -> (M[pos[k], j] += c), t, b)
    end
    return M
end

# Operator-level action using PauliOperators algebra (Schrödinger and Heisenberg pictures)
function Base.:*(Λ::Lindbladian, ρ)
    dρ = -1im * (Λ.H * ρ - ρ * Λ.H)
    for (Lk, γ) in zip(Λ.L, Λ.γ)
        LL = Lk' * Lk
        dρ += γ * ((Lk * ρ) * Lk')
        dρ += -0.5γ * (LL * ρ + ρ * LL)
    end
    return dρ
end

Base.adjoint(Λ::Lindbladian) = Adjoint(Λ)   # L' acts in the Heisenberg picture

function Base.:*(Λa::Adjoint{<:Any,<:Lindbladian}, O)
    Λ = parent(Λa)
    dO = 1im * (Λ.H * O - O * Λ.H)
    for (Lk, γ) in zip(Λ.L, Λ.γ)
        LL = Lk' * Lk
        dO += γ * (Lk' * O * Lk)
        dO += -0.5γ * (LL * O + O * LL)
    end
    return dO
end
