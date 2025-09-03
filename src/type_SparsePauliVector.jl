using OrderedCollections


SparsePauliVectors{N,T} = OrderedDict{PauliBasis{N}, Vector{T}} 
Base.adjoint(d::SparsePauliVectors{N,T}) where {N,T} = Adjoint(d)

function SparsePauliVectors(ps::PauliSum{N,T}; R=1) where {N,T}
    spv = OrderedDict{PauliBasis{N}, Vector{T}}()
    for (pauli,coeff) in ps
        spv[pauli] = [coeff] 
        for s in 2:R
            push!(spv[pauli],0)
        end
    end
    return spv
end

function clip!(ps::SparsePauliVectors{N,T}; thresh=1e-16) where {N,T}
    filter!(p->maximum(abs.(p.second)) > thresh, ps)
end

Base.Vector(spv::SparsePauliVectors{N,T}; state=1)  where {N,T} = [i[state] for i in values(spv)]
function Base.Matrix(spv::SparsePauliVectors{N,T})  where {N,T}
    ni, nj = size(spv)
    out = zeros(T, ni, nj)
    i = 1
    for (d,coeffs) in spv
        for j in 1:nj 
            out[i,j] = coeffs[j]
        end
        i += 1
    end
    return out
end

function Base.size(spv::SparsePauliVectors)
    ni = length(spv)
    for (d,c) in spv
        return (ni, length(c)) 
    end
end

function Base.sum!(spv::SparsePauliVectors{N,T}, pauli::PauliBasis{N}, coeffs::Vector{T}) where {N,T}
    if haskey(spv, pauli)
        spv[pauli] .+= coeffs
    else
        spv[pauli] = coeffs
    end
end

function todense(spv::SparsePauliVectors{N,T}) where {N,T}
    out = zeros(T, 4^N, size(spv)[2])
    for (d,c) in spv
        out[index(d),:] .= c
    end
    return out
end

# function Base.display(ps::SparsePauliVectors)
#     for (key,val) in ps
#         @printf(" %12s ", key)
#         for s in 1:length(val)
#             @printf(" %12.8f +%12.8fi", real(val[s]), imag(val[s]))
#         end
#         @printf("\n")
#     end
# end

function Base.display(ps::SparsePauliVectors)
    for (key, val) in ps
        @printf("%12s ", string(key))  # Ensure key is converted to string for proper formatting
        for s in eachindex(val)
            @printf("%12.8f + %12.8fi ", real(val[s]), imag(val[s]))
        end
        println()  # Move to the next line for better readability
    end
end

# function Base.display(ps::SparsePauliVectors; state=1)
#     for (key,val) in ps
#         @printf(" %12s ", key)
#         @printf(" %12.8f +%12.8fi", real(val[state]), imag(val[state]))
#         @printf("\n")
#     end
# end

function PauliSum(spv::SparsePauliVectors{N,T}; state=1) where {N,T} 
    out = PauliSum(N,T)
    for (d,c) in spv
        out[d] = c[state]
    end
    return out
end

function eye!(spv::SparsePauliVectors{N,T}) where {N,T}
    R = 1
    for (d,c) in spv
        R = length(c)
        break
    end
    if size(spv)[1] < size(spv)[2]
        throw(DimensionMismatch)
    end
    Imat = Matrix(T(1)*I, size(spv)...)
    fill!(spv, Imat[:,1:R])
end

function Base.:*(a::Adjoint{<:Any, SparsePauliVectors{N,T}}, b::SparsePauliVectors{N,T}) where {N,T}
    out = zeros(T,size(a.parent)[2], size(b)[2]) 
    for (da,ca) in a.parent
        if haskey(b,da)
            out .+= conj(ca)*transpose(b[da])
        end
    end
    return out
end