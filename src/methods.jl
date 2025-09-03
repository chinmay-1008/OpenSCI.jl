using PauliOperators
using OpenSCI


"""
    Base.:*(Lprime::Adjoint{<:Any, Lindbladian{N}}, A) where {N}

Multiplication of L' with A, which is either a PauliSum or a DyadSum.
The adjoint here is generally used for performing Heisenberg evolution.
"""
function Base.:*(Lprime::Adjoint{<:Any, Lindbladian{N}}, A) where {N}
    L = Lprime.parent
    dA = 1im * (L.H*A - A*L.H)
    for i in 1:length(L.γ)
        Li = L.L[i]
        dA += L.γ[i] * (Li' * A * Li)
        LL = Li' * Li
        dA -= 0.5*L.γ[i]*(LL * A + A * LL)
    end
    return dA 
end

"""
    Base.:*(L::Lindbladian{N}, ρ) where {N}

Multiplication of L with A, which is either a PauliSum or a DyadSum
"""
function Base.:*(L::Lindbladian{N}, ρ) where {N}
    dρ = -1im * (L.H*ρ - ρ*L.H)
    for i in 1:length(L.γ)
        Li = L.L[i]
        dρ += L.γ[i] * ((Li * ρ) * Li')
        LL = Li' * Li
        dρ += -0.5*L.γ[i]*(LL * ρ + ρ * LL)
    end
    return dρ 
end
# function Base.:*(L::Lindbladian{N}, ρ::DyadSum{N,T}) where {N}
#     dρ = -1im * (L.H*ρ - ρ*L.H)
#     for i in 1:length(L.γ)
#         Li = L.L[i]
#         dρ += L.γ[i] * (Li * ρ * Li')
#         LL = Li' * Li
#         dρ -= 0.5*L.γ[i]*(LL * ρ + ρ * LL)
#     end
#     return dρ 
# end


"""
    Dyad2Pauli_rotation(N)

Compute the rotation matrix to go from the Dyad (standard) basis
to the Pauli basis.
U(ij,k) = tr(|i><j| * Pk)
# Arguments
- `N::Integer`: the number of qubits
"""
function Dyad2Pauli_rotation(N)
    dim = 2^N
    U = zeros(ComplexF64, (4^N, 4^N))
    normalization = sqrt(dim)
    for k in 0:dim-1 
        for b in 0:dim-1 
            d = Dyad(N,k,b)
            for z in 0:dim-1 
                for x in 0:dim-1 
                    p = Pauli(z,x, N)
                    U[index(d), index(p)] = tr(d*p)/normalization
                end
            end
        end
    end
    return U
end

"""
    Base.Matrix(L::Lindbladian{N}) where N

Build the matrix representation of `L` in the standard (`Dyad`) basis
# Arguments
- `L::Lindbladian`: The Lindbladian 
"""
function Base.Matrix(L::Lindbladian{N}) where N
    Lmat = zeros(ComplexF64, 4^N, 4^N)
    for kr in 0:2^N-1
        for br in 0:2^N-1
            dyad_r = Dyad(N, kr, br)
            
            σ = L * dyad_r        
            for (dyad_l, coeff) in σ
                Lmat[index(dyad_l), index(dyad_r)] = coeff
            end
        end
    end
    return Lmat
end

Base.Matrix(Lprime::Adjoint{<:Any, Lindbladian}) = Matrix(L.parent)'

function add_hamiltonian!(L::Lindbladian{N}, H::PauliSum{N}) where N
    sum!(L.H, H)
    return L
end

function add_channel_amplitude_damping!(L::Lindbladian{N}, γ::Real) where N
    return add_channel_amplitude_damping!(L, [γ for i in 1:N])
end

function add_channel_dephasing!(L::Lindbladian{N}, γ::Real) where N
    return add_channel_dephasing!(L, [γ for i in 1:N])
end

function add_channel_depolarizing!(L::Lindbladian{N}, γ::Real) where N
    return add_channel_depolarizing!(L, [γ for i in 1:N])
end


function add_channel_amplitude_damping!(L::Lindbladian{N}, γ::Vector{<:Real}) where N
    length(γ) == N || throw(ArgumentError("Length of γ must be equal to N"))
    for i in 1:N
        pi = PauliSum(N)
        pi += Pauli(N, X=[i]) + 1im * Pauli(N, Y=[i])
        # pi += 1/sqrt(2)*(Pauli(N, X=[i]) + 1im * Pauli(N, Y=[i]))
        push!(L.L, pi)
        push!(L.γ, γ[i])
    end
end


function add_channel_dephasing!(L::Lindbladian{N}, γ::Vector{<:Real}) where N
    length(γ) == N || throw(ArgumentError("Length of γ must be equal to N"))
    for i in 1:N
        pi = PauliSum(N)
        pi += 1*Pauli(N, Z=[i])
        push!(L.L, pi)
        push!(L.γ, γ[i])
    end
end


function add_channel_depolarizing!(L::Lindbladian{N}, γ::Vector{<:Real}) where N
    length(γ) == N || throw(ArgumentError("Length of γ must be equal to N"))
    for i in 1:N
        pi = PauliSum(N)
        pi += Pauli(N, X=[i])
        push!(L.L, pi)
        push!(L.γ, γ[i])
        
        pi = PauliSum(N)
        pi += Pauli(N, Y=[i])
        push!(L.L, pi)
        push!(L.γ, γ[i])
        
        pi = PauliSum(N)
        pi += Pauli(N, Z=[i])
        push!(L.L, pi)
        push!(L.γ, γ[i])
    end
end

Base.exp(D::WeightDissipator{N}) where N = WeightDissipator{N}(D.l, exp(D.γ))
Base.exp(D::SubspaceDissipator{N}) where N = SubspaceDissipator{N}(D.set, exp(D.γ))
Base.:*(D::WeightDissipator{N}, a::Number) where N = WeightDissipator{N}(D.l, D.γ*a)
Base.:*(D::SubspaceDissipator{N}, a::Number) where N = WeightDissipator{N}(D.set, D.γ*a)
Base.:*(a::Number, D::WeightDissipator) = D*a 
Base.:*(a::Number, D::SubspaceDissipator) = D*a 

function pauli_weight(p::PauliBasis) 
    return count_ones(p.z | p.x)
end


function LinearAlgebra.Diagonal(D::WeightDissipator{N}; T=Float64) where N
    diag = ones(T, (4^N))
    for z in 0:2^N-1
        for x in 0:2^N-1
            pi = Pauli(PauliBasis(z,x))
            if pauli_weight(pi) > D.l
                diag[index(pi)] = D.γ
            end
        end
    end
    return Diagonal(diag) 
end

function LinearAlgebra.Diagonal(D::SubspaceDissipator{N}; T=Float64) where N
    diag = ones(T, (4^N))
    for z in 0:2^N-1
        for x in 0:2^N-1
            pi = PauliBasis{N}(z,x)
            if haskey(D.set, pi) == false
                diag[index(pi)] = D.γ
            end
        end
    end
    return Diagonal(diag) 
end

function commute(p1::Union{Pauli{N},PauliBasis{N}}, p2::Union{Pauli{N},PauliBasis{N}}) where N 
    return iseven(count_ones(p1.x & p2.z) - count_ones(p1.z & p2.x))
end 

"""
    evolve(P::PauliSum{N,T}, G::Pauli{N}) where {N,T}

Evolve `P` by Pauli `G`
    exp(i G/2) P exp(-i G/2) = cos(α)P + isin(α)2*P*G if [P,G] ≠ 0
    
where α=coeff(G)
"""
function evolve(P::PauliSum{N,T}, G::Pauli{N}) where {N,T}
    _cos = cos(coeff(G))
    _sin = -1im*sin(coeff(G))
    out = deepcopy(P) 
    sin_branch = PauliSum(N)
    for (p,c) in P
        if commute(p,G) == false
            out[p] *= _cos
            sum!(sin_branch, c*_sin*p*PauliBasis(G))
        end
    end
    sum!(out, sin_branch)
    return out
end