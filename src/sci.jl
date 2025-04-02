using PauliOperators
using OpenSCI
using Arpack

function selected_ci(L::Lindbladian{N}, v::SparseDyadVectors{N,T}; 
    ϵsearch=1e-1, ϵdiscard=1e-4, max_iter_outer=4, thresh_conv=1e-5,
    verbose = 0) where {N,T}

    dim, R = size(v)
    # display(v)
    # @show size(v)
    last = deepcopy(v)

    Pv= deepcopy(v)
    
    for n_iter in 1:max_iter_outer
        # @printf("\n")
        # @printf("\n ####################################")
        # @printf("\n SCI Iteration: %4i\n", n_iter)
        # display(Pv)
        clip!(Pv, thresh=ϵdiscard) 
        # println("clipclipclip")
        # display(Pv)
        σ = multiply(L, Pv, ϵ=ϵsearch)
        # println("prprprprprprp")
        # display(σ)


        for (d,c) in σ
            if maximum(abs2.(c)) > ϵdiscard
                # sum!(PLP, d, c)
                sum!(Pv, d, zeros(T, R))
            else
                # sum!(QLP, d, c)
            end
        end
        # display(Pv)
        # display(size(Pv))

        Lmat = build_subspace_L(L, Pv)
        e = 0
        v = zeros(T,size(Pv))
        # println(size(Lmat)," ", size(L))
        # display(Lmat)
        if length(Pv) < 300
            e,v = eigen(Lmat)
            # println("VEC")
            # display(v)
            e = e[end-R+1:end]
            # println("MOD VEC")
            v = v[:, end-R+1:end]
            # display(v)

        else
            e,v = eigs(Lmat, nev=R, v0=Matrix(Pv)[:,1], which=:LR, maxiter=500)
        end
        # println("\nState")
        # display(Pv)
        # println("\n Fill State")
        fill!(Pv, v)
        # display(Pv)
        if verbose > 1
            display(Pv'*last)
        end

        ovlap = Pv'*last
        # display(Lmat)

        # @printf("\n Eigenvalues of Lmat:\n")
        # display(e)
        # println("############")
        # for i in eachindex(e)
            # @printf(" %4i % 12.8f % 12.8fi Δ = %12.8f\n", i, real(e[i]), imag(e[i]), abs(ovlap[i,i]))
        # end


        if length(Pv) == length(last)
            tmp1 = Matrix(Pv)

            tmp2 = Matrix(last)
            ovlap = pinv(tmp1)*tmp2
            # @show det(ovlap) 
            if abs(1 - det(ovlap)) < thresh_conv 
                @printf(" *Converged\n")
                break
            end
        end
        
        last = deepcopy(Pv)
        # println("\nMAT Pv")
        # display(Matrix(Pv))
    end

    return Pv
end


function multiply(L::Lindbladian{N}, ρ::SparseDyadVectors{N,T}; ϵ=1e-16) where {N,T}
    σ = SparseDyadVectors{N,T}()

    # Unitary part
    for (rdyad, rcoeffs) in ρ

        σi = -1im * (L.H * rdyad - rdyad * L.H)
        
        for i in 1:length(L.γ)
            Li = L.L[i]
            LL = Li' * Li
           
            σi +=  L.γ[i] * (Li * rdyad * Li')
            σi -= 0.5*L.γ[i]*(LL*rdyad + rdyad*LL)
        end
        
        for (ldyad,lcoeff) in σi 
            sum!(σ, ldyad, lcoeff .* rcoeffs )
        end
    end
    return σ 
end

function Base.:*(L::Lindbladian{N}, ρ::SparseDyadVectors{N,T}) where {N,T}
    return multiply(L,ρ)
end

function build_subspace_L(L::Lindbladian{N}, v::SparseDyadVectors{N,T}) where {N,T}
   
    indices = Dict{DyadBasis{N}, Int}()
    idx = 1
    for (d,c) in v
        indices[d] = idx
        idx += 1
    end

    dim = length(v)
    
    Lmat = zeros(T, dim, dim)

    vi = 0
    # Unitary part
    for rdyad in keys(v)
        vi = indices[rdyad] 

        for (pauli, coefficient) in L.H
            
            ldyad = coefficient * (pauli * rdyad)
            if haskey(v, DyadBasis(ldyad)) 
                Lmat[indices[DyadBasis(ldyad)], vi] += -1im * coeff(ldyad)
            end
            
            ldyad = coefficient * (rdyad * pauli)
            if haskey(v, DyadBasis(ldyad)) 
                Lmat[indices[DyadBasis(ldyad)], vi] += 1im * coeff(ldyad)
            end
        end

        
        for i in 1:length(L.γ)
            γ = L.γ[i]
            for (pj, cj) in L.L[i]
                for (pk, ck) in L.L[i]
                    Pj = pj * cj
                    Pk = pk * ck

                    ldyad = γ * (Pj * rdyad * Pk')
                    if haskey(v, DyadBasis(ldyad)) 
                        Lmat[indices[DyadBasis(ldyad)], vi] += coeff(ldyad)
                    end
                    
                    ldyad = γ * (rdyad * Pj' * Pk)
                    if haskey(v, DyadBasis(ldyad)) 
                        Lmat[indices[DyadBasis(ldyad)], vi] -= .5 * coeff(ldyad)
                    end
                    
                    ldyad = γ * (Pj' * Pk * rdyad)
                    if haskey(v, DyadBasis(ldyad)) 
                        Lmat[indices[DyadBasis(ldyad)], vi] -= .5 * coeff(ldyad)
                    end
                end
            end
        end
    end
    return Lmat 
end


function Base.fill!(sdv::SparseDyadVectors{N,T}, m::Matrix{T}) where {N,T}
    size(sdv) == size(m) || throw(DimensionMismatch)

    ridx = 0
    for (d,coeffs) in sdv
        ridx += 1
        sdv[d] .= m[ridx,:]
    end
    return sdv
end

function PauliOperators.clip!(sdv::SparseDyadVectors; thresh=1e-5)
    filter!(p->maximum(abs2.(p.second)) > thresh, sdv)
    return sdv
end


# function sparse_lindbladian_eigensolve(L::Lindbladian, v0::SparseDyadVectors)
#     Lmat = Matrix(L)
#     l = eigvals(Lmat)
#     @printf("\n Eigenvalues of Lmat:\n")
#     for i in 1:length(l)
#         @printf(" %4i % 12.8f % 12.8fi\n", i, real(l[i]), imag(l[i]))
#     end
# end