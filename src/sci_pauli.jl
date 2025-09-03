using PauliOperators
using OpenSCI
using Arpack

function selected_ci(L::Lindbladian{N}, v::SparsePauliVectors{N,T}; 
    ϵsearch=1e-1, ϵdiscard=1e-4, max_iter_outer=4, thresh_conv=1e-7,
    verbose = 0) where {N,T}

    dim, R = size(v)
    # display(v)
    # @show size(v)
    last = deepcopy(v)

    Pv= deepcopy(v)
    e = 0
    for n_iter in 1:max_iter_outer
        @printf("\n ####################################")
        @printf("\n SCI Iteration: %4i \n", n_iter)
        clip!(Pv, thresh=ϵdiscard)

        display("P---PAULI")
        display(size(Pv))

        σ = multiply(L, Pv, ϵ=ϵsearch)

        for (d,c) in σ
            if maximum(abs2.(c)) > ϵdiscard
                sum!(Pv, d, zeros(T, R))
            else
            end
        end

        println("NEW-PPAULI")
        display(size(Pv))
        Lmat = build_subspace_L(L, Pv)
        e = 0
        v = zeros(T,size(Pv))
        # println()
        # display(size(Pv))
        if length(Pv) < 3000
            e,v = eigen(Lmat)
            # e = e[end-R+1:end]
            # v = v[:, end-R+1:end]
            perm = sortperm(real(e))
            e = e[perm[end-R+1:end]]
            v = v[:, perm[end-R+1:end]]
            
            println("Eigen")
        else
            e,v = eigs(Lmat, nev=R, v0=Matrix(Pv)[:,1], which=:LR, maxiter=3000 , tol=1e-5, check=1)
            println("Eigs")
            perm = sortperm(real(e))
            e = e[perm]
            v = v[:, perm]
        end
        
        
        fill!(Pv, v)
        if verbose > 1
            display(Pv'*last)
        end

        ovlap = Pv'*last

        # @printf("Eigenvalues of Lmat:\n")
        # # display(e)
        # # println("############")
        # for i in eachindex(e)
        #     @printf(" %4i % 12.8f % 12.8fi Δ = %12.8f\n", i, real(e[i]), imag(e[i]), abs(ovlap[i,i]))
        # end


        if length(Pv) == length(last)
            tmp1 = Matrix(Pv)

            tmp2 = Matrix(last)
            ovlap = pinv(tmp1)*tmp2
            if abs(1 - det(ovlap)) < thresh_conv 
                @printf(" *Converged\n")
                break
            end
        end
        
        last = deepcopy(Pv)
    end
    return Pv, e
end


function prepare_lindblad_cache(L::Lindbladian{N}) where {N}
    return [(Li, Li' * Li) for Li in L.L]
end

function multiply(L::Lindbladian{N}, ρ::SparsePauliVectors{N,T}; ϵ=1e-16) where {N,T}
    σ = SparsePauliVectors{N,T}()

    L_dat = prepare_lindblad_cache(L)
    # Unitary part
    for (rpauli, rcoeffs) in ρ
        σi = PauliSum(N)
        σi += -1im * (L.H * rpauli - rpauli * L.H)

        for i in 1:length(L.γ)
            # Li = L.L[i]
            # LL = Li' * Li
            Li, LL = L_dat[i]
            
            σi +=  L.γ[i] * (Li * rpauli * Li')
            σi -= 0.5*L.γ[i]*(LL*rpauli + rpauli*LL)
        end
        
        for (lpauli,lcoeff) in σi 
            sum!(σ, lpauli, lcoeff .* rcoeffs )
        end
    end
    return σ 
end


function selected_ci_first_correction(L::Lindbladian{N}, v::SparsePauliVectors{N,T}; 
    ϵsearch=1e-1, ϵdiscard=1e-4, max_iter_outer=4, thresh_conv=1e-7,
    verbose = 0) where {N,T}

    dim, R = size(v)
    # display(v)
    # @show size(v)
    last = deepcopy(v)

    Pv= deepcopy(v)
    e = 0
    # v = 
    for n_iter in 1:max_iter_outer
        @printf("\n ####################################")
        @printf("\n SCI Iteration: %4i \n", n_iter)
        clip!(Pv, thresh=ϵdiscard)

        display("P---DYAD")
        display(size(Pv))
        σ = multiply(L, Pv, ϵ=ϵsearch)

        v = zeros(T,size(Pv))

        x_dyad = SparsePauliVectors(PauliSum(N), R=R)
        # This represents |q>><<q|L|v_i>>
        for (d, coeff) in σ
            if !haskey(Pv, d)
                sum!(x_dyad, d, coeff)
            end
        end

        if n_iter > 1
            for (d, c) in x_dyad
                one_x_dyad = SparsePauliVectors(PauliSum(d), R = R)
                one_x_dyad[d] = c
                temp = L * one_x_dyad

                Lxx = conj.(c) .* temp[d]

                x_dyad[d] = c ./ (Lxx .- e )

            end
        end

        for (d,c) in x_dyad
            if maximum(abs2.(c)) > ϵdiscard
                sum!(Pv, d, zeros(T, R))
            else
            end
        end
        
        println("NEW-PDYAD")
        display(size(Pv))
        Lmat = build_subspace_L(L, Pv)

        # println()
        # display(size(Pv))
        if length(Pv) < 3000
            e,v = eigen(Lmat)
            # e = e[end-R+1:end]
            # v = v[:, end-R+1:end]
            perm = sortperm(real(e))
            e = e[perm[end-R+1:end]]
            v = v[:, perm[end-R+1:end]]
            
            println("Eigen")
        else
            e,v = eigs(Lmat, nev=R, v0=Matrix(Pv)[:,1], which=:LR, maxiter=3000 , tol=1e-5, check=1)
            println("Eigs")
            perm = sortperm(real(e))
            e = e[perm]
            v = v[:, perm]
        end
        
        
        fill!(Pv, v)
        if verbose > 1
            display(Pv'*last)
        end

        ovlap = Pv'*last

        # @printf("Eigenvalues of Lmat:\n")
        # # display(e)
        # # println("############")
        # for i in eachindex(e)
        #     @printf(" %4i % 12.8f % 12.8fi Δ = %12.8f\n", i, real(e[i]), imag(e[i]), abs(ovlap[i,i]))
        # end


        if length(Pv) == length(last)
            tmp1 = Matrix(Pv)

            tmp2 = Matrix(last)
            ovlap = pinv(tmp1)*tmp2
            if abs(1 - det(ovlap)) < thresh_conv 
                @printf(" *Converged\n")
                break
            end
        end
        
        last = deepcopy(Pv)
    end
    return Pv, e
end


function Base.:*(L::Lindbladian{N}, ρ::SparsePauliVectors{N,T}) where {N,T}
    return multiply(L,ρ)
end

function build_subspace_L(L::Lindbladian{N}, v::SparsePauliVectors{N,T}) where {N,T}
   
    indices = Dict{PauliBasis{N}, Int}()
    idx = 1
    for (d,c) in v
        indices[d] = idx
        idx += 1
    end

    dim = length(v)
    
    Lmat = zeros(T, dim, dim)

    vi = 0
    # Unitary part
    for rpauli in keys(v)
        vi = indices[rpauli] 

        for (pauli, coefficient) in L.H
            
            lpauli = coefficient * (pauli * rpauli)
            if haskey(v, PauliBasis(lpauli)) 
                Lmat[indices[PauliBasis(lpauli)], vi] += -1im * coeff(lpauli)
            end
            
            lpauli = coefficient * (rpauli * pauli)
            if haskey(v, PauliBasis(lpauli)) 
                Lmat[indices[PauliBasis(lpauli)], vi] += 1im * coeff(lpauli)
            end
        end

        
        for i in 1:length(L.γ)
            γ = L.γ[i]
            for (pj, cj) in L.L[i]
                for (pk, ck) in L.L[i]
                    Pj = pj * cj
                    Pk = pk * ck

                    lpauli = γ * (Pj * rpauli * Pk')
                    if haskey(v, PauliBasis(lpauli)) 
                        Lmat[indices[PauliBasis(lpauli)], vi] += coeff(lpauli)
                    end
                    
                    lpauli = γ * (rpauli * Pj' * Pk)
                    if haskey(v, PauliBasis(lpauli)) 
                        Lmat[indices[PauliBasis(lpauli)], vi] -= .5 * coeff(lpauli)
                    end
                    
                    lpauli = γ * (Pj' * Pk * rpauli)
                    if haskey(v, PauliBasis(lpauli)) 
                        Lmat[indices[PauliBasis(lpauli)], vi] -= .5 * coeff(lpauli)
                    end
                end
            end
        end
    end
    return Lmat 
end

function build_subspace_L_generalized(L::Lindbladian{N}, v_row::SparsePauliVectors{N,T}, v_col::SparsePauliVectors{N, T}) where {N,T}
    row_indices = Dict{PauliBasis{N}, Int}()
    col_indices = Dict{PauliBasis{N}, Int}()

    for (i, (d,c)) in enumerate(v_row)
        row_indices[d] = i
    end

    for (j, (d, c)) in enumerate(v_col)
        col_indices[d] = j
    end

    row_dim = length(v_row)
    col_dim = length(v_col)
    
    Lmat = zeros(T, row_dim, col_dim)

    vi = 0
    # Unitary part
    for rpauli in keys(v_col)
        vi = col_indices[rpauli] 

        for (pauli, coefficient) in L.H
            
            lpauli = coefficient * (pauli * rpauli)
            if haskey(row_indices, PauliBasis(lpauli)) 
                Lmat[row_indices[PauliBasis(lpauli)], vi] += -1im * coeff(lpauli)
            end
            
            lpauli = coefficient * (rpauli * pauli)
            if haskey(row_indices, PauliBasis(lpauli)) 
                Lmat[row_indices[PauliBasis(lpauli)], vi] += 1im * coeff(lpauli)
            end
        end

        # Dissipation part
        for i in 1:length(L.γ)
            γ = L.γ[i]
            for (pj, cj) in L.L[i]
                for (pk, ck) in L.L[i]
                    Pj = pj * cj
                    Pk = pk * ck

                    lpauli = γ * (Pj * rpauli * Pk')
                    if haskey(row_indices, PauliBasis(lpauli)) 
                        Lmat[row_indices[PauliBasis(lpauli)], vi] += coeff(lpauli)
                    end
                    
                    lpauli = γ * (rpauli * Pj' * Pk)
                    if haskey(row_indices, PauliBasis(lpauli)) 
                        Lmat[row_indices[PauliBasis(lpauli)], vi] -= .5 * coeff(lpauli)
                    end
                    
                    lpauli = γ * (Pj' * Pk * rpauli)
                    if haskey(row_indices, PauliBasis(lpauli)) 
                        Lmat[row_indices[PauliBasis(lpauli)], vi] -= .5 * coeff(lpauli)
                    end
                end
            end
        end
    end
    return Lmat 
end


function Base.fill!(sdv::SparsePauliVectors{N,T}, m::Matrix{T}) where {N,T}
    size(sdv) == size(m) || throw(DimensionMismatch)

    ridx = 0
    for (d,coeffs) in sdv
        ridx += 1
        sdv[d] .= m[ridx,:]
    end
    return sdv
end

function PauliOperators.clip!(sdv::SparsePauliVectors; thresh=1e-5)
    filter!(p->maximum(abs2.(p.second)) > thresh, sdv)
    return sdv
end


# function sparse_lindbladian_eigensolve(L::Lindbladian, v0::SparsePauliVectors)
#     Lmat = Matrix(L)
#     l = eigvals(Lmat)
#     @printf("\n Eigenvalues of Lmat:\n")
#     for i in 1:length(l)
#         @printf(" %4i % 12.8f % 12.8fi\n", i, real(l[i]), imag(l[i]))
#     end
# end