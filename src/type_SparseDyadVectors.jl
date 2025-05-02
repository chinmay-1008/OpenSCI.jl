using OrderedCollections


SparseDyadVectors{N,T} = OrderedDict{DyadBasis{N}, Vector{T}} 
Base.adjoint(d::SparseDyadVectors{N,T}) where {N,T} = Adjoint(d)

function SparseDyadVectors(ds::DyadSum{N,T}; R=1) where {N,T}
    sdv = OrderedDict{DyadBasis{N}, Vector{T}}()
    for (dyad,coeff) in ds
        sdv[dyad] = [coeff] 
        for s in 2:R
            push!(sdv[dyad],0)
        end
    end
    return sdv
end

function clip!(ps::SparseDyadVectors{N,T}; thresh=1e-16) where {N,T}
    filter!(p->maximum(abs.(p.second)) > thresh, ps)
end

Base.Vector(sdv::SparseDyadVectors{N,T}; state=1)  where {N,T} = [i[state] for i in values(sdv)]
function Base.Matrix(sdv::SparseDyadVectors{N,T})  where {N,T}
    ni, nj = size(sdv)
    out = zeros(T, ni, nj)
    i = 1
    for (d,coeffs) in sdv
        for j in 1:nj 
            out[i,j] = coeffs[j]
        end
        i += 1
    end
    return out
end

function Base.size(sdv::SparseDyadVectors)
    ni = length(sdv)
    for (d,c) in sdv
        return (ni, length(c)) 
    end
end

function Base.sum!(sdv::SparseDyadVectors{N,T}, dyad::DyadBasis{N}, coeffs::Vector{T}) where {N,T}
    if haskey(sdv, dyad)
        sdv[dyad] .+= coeffs
    else
        sdv[dyad] = coeffs
    end
end

function todense(sdv::SparseDyadVectors{N,T}) where {N,T}
    out = zeros(T, 4^N, size(sdv)[2])
    for (d,c) in sdv
        out[index(d),:] .= c
    end
    return out
end

# function Base.display(ps::SparseDyadVectors)
#     for (key,val) in ps
#         @printf(" %12s ", key)
#         for s in 1:length(val)
#             @printf(" %12.8f +%12.8fi", real(val[s]), imag(val[s]))
#         end
#         @printf("\n")
#     end
# end

function Base.display(ps::SparseDyadVectors)
    for (key, val) in ps
        @printf("%12s ", string(key))  # Ensure key is converted to string for proper formatting
        for s in eachindex(val)
            @printf("%12.8f + %12.8fi ", real(val[s]), imag(val[s]))
        end
        println()  # Move to the next line for better readability
    end
end

# function Base.display(ps::SparseDyadVectors; state=1)
#     for (key,val) in ps
#         @printf(" %12s ", key)
#         @printf(" %12.8f +%12.8fi", real(val[state]), imag(val[state]))
#         @printf("\n")
#     end
# end

function DyadSum(sdv::SparseDyadVectors{N,T}; state=1) where {N,T} 
    out = DyadSum(N,T)
    for (d,c) in sdv
        out[d] = c[state]
    end
    return out
end

function eye!(sdv::SparseDyadVectors{N,T}) where {N,T}
    R = 1
    for (d,c) in sdv
        R = length(c)
        break
    end
    if size(sdv)[1] < size(sdv)[2]
        throw(DimensionMismatch)
    end
    Imat = Matrix(T(1)*I, size(sdv)...)
    fill!(sdv, Imat[:,1:R])
end

function Base.:*(a::Adjoint{<:Any, SparseDyadVectors{N,T}}, b::SparseDyadVectors{N,T}) where {N,T}
    out = zeros(T,size(a.parent)[2], size(b)[2]) 
    for (da,ca) in a.parent
        if haskey(b,da)
            out .+= conj(ca)*transpose(b[da])
        end
    end
    return out
end