using OpenSCI
using PauliOperators
using LinearAlgebra
using DifferentialEquations
using Plots, Measures
using Printf
using OrderedCollections
using StatProfilerHTML
gr(display_type=:inline)


function plot_population_lowdin_eigen(λ, V, V_P, ρ0, times, j)

    D, R = size(V_P)
    d = round(Int, sqrt(D))  

    ρ0_P = ρ0               

    W = inv(V) 

    pops = zeros(length(times), d)

    for (ti, t) in enumerate(times)
        exp_diag = Diagonal(exp.(λ * t))         
        ρp_t = V * exp_diag * W * ρ0_P           
        ρ_vec_t = V_P * ρp_t                      
        ρ_mat_t = reshape(ρ_vec_t, d, d)          
        pops[ti, :] .= real(diag(ρ_mat_t))         
    end

    # labels = ["|$(i)⟩" for i in 0:d-1]
    labels = reduce(hcat, [["|$(i)⟩"] for i in 0:d-1])
    plot(times, pops, xlabel="Time", ylabel="Population", label=labels, ylim = (-0.2, 1), size=(600, 400), top_margin=5mm,bottom_margin = 5mm, right_margin=5mm, left_margin=5mm, dpi=300,)
    N = log2(d)
    savefig("test/lowdin_$N-$R-$j.pdf")

end


# display(ω + v_p[:, 1]' * L_PQ * inv(ω * I - L_QQ) * L_QP * v_p[:, 1])
# println("+++++++++++++++++++++++")
# ρ_P = v_p' * vec_state_i
# # display(ρ_P)
# time_step = [i/10 for i in 0:50]
# plot_population_lowdin_eigen(e, v, v_p, ρ_P, time_step, count )
# count+=1
# end
function sparse_vec(v::SparseDyadVectors{N, T}) where {N, T}
    indices = Dict{DyadBasis{N}, Int}()
    idx = 1
    for (d, _) in v
        indices[d] = idx
        idx += 1
    end

    dim, R = size(v)
    sparse_v = zeros(ComplexF64, dim, R)

    for (d, c) in v
        row = indices[d]
        sparse_v[row, :] .= c
    end

    return sparse_v
end



function matrix_run()
    N = 6
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)  
    add_channel_depolarizing!(L, .1)
    println("Lindbladian: ")
    display(L)

    Lmat = Matrix(L)
    dim_L = size(Lmat, 1)
    # println("Matrix Form of L: ")
    # println("Diagonalization started")

    F = eigen(Lmat)
    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]
    @printf("\n Eigenvalues of L:\n")
    for i in (length(F.values) - 2):length(F.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states[i])))
    end

    display(size(Lmat))

    state = DyadSum(Dyad(N, 0, 0))
    # vec_state_i = vec(Matrix(state))
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)

    p_dyad, eig_sci = selected_ci(L, v0, max_iter_outer = 5)
    println("\n Eigenvalues after SCI")
    display(eig_sci)

    vec_p_dyad = sparse_vec(p_dyad)
    # display(vec_p_dyad)
    # return
    lmat_pp = build_subspace_L(L, p_dyad)
    # println("P-Space")
    # display(lmat_pp)
    x_temp = L*p_dyad
    # println("P-Space")
    # display(state_sci)



    x_dyad = SparseDyadVectors(DyadSum(N), R=nkeep)

    for (d, coeff) in x_temp
        if !haskey(p_dyad, d)
            sum!(x_dyad, d, coeff)
        end
    end 

    lmat_xx = build_subspace_L(L, x_dyad)

    lmat_px = build_subspace_L_generalized(L, p_dyad, x_dyad)
    
    lmat_xp = build_subspace_L_generalized(L, x_dyad, p_dyad)

    n = size(lmat_xx)[1]
    identity_matrix = Matrix{Float64}(I, n, n)

    corr_idx = 2
    lambda = eig_sci[corr_idx]*identity_matrix

    pertubation  = lmat_px * inv(lambda - lmat_xx) * lmat_xp
    leff = lmat_pp + pertubation
    # println("Pertubation")
    # display(pertubation)

    e, v = eigen(leff)
    println("\n Eigenvalues after Lowdin")
    new_eig = e[end-nkeep+1:end]
    display(new_eig)

    println("\n Using SCI eigenvector")

    sci_vec = vec_p_dyad[:, corr_idx]' * leff * vec_p_dyad[:, corr_idx]
    display(sci_vec)
    # println("temp")
    # display(lmat_px)
    # println("Difference between the eigenvalue")
    # display(abs.(eig_sci - new_eig))
    return
end

function left_eigenvectors(dyad_dict::SparseDyadVectors{N,T}, Lmat)::SparseDyadVectors{N,T} where {N,T}
    # dyad_keys = collect(keys(dyad_dict))                            
    # values_matrix = transpose(hcat(values(dyad_dict)...))           
    e = 0
    v = zeros(T,size(dyad_dict))
    display(size(dyad_dict))
    Lmat = Lmat'
    dim, R = size(dyad_dict)
    if length(dyad_dict) < 300
        e,v = eigen(Lmat)
        e = e[end-R+1:end]
        v = v[:, end-R+1:end]
        
        println("Eigen for left")
        display(e)
    else
        e,v = eigs(Lmat, nev=R, v0=Matrix(Pv)[:,1], which=:LR, maxiter=3000 , tol=1e-5, check=1)
        println("Eigs")
        perm = sortperm(real(e))
        e = e[perm]
        v = v[:, perm]
    end
    
    # perm = sortperm(real(e))
    # e = e[perm[end-R+1:end]]
    # v = v[:, perm[end-R+1:end]]
    
    OpenSCI.fill!(dyad_dict, v)
    # pinv_matrix = pinv(values_matrix)                              
    # dim, R = size(dyad_dict)
    # pinv_sdv = OrderedDict{DyadBasis{N}, Vector{T}}()
    # pinv_sdv = SparseDyadVectors(DyadSum(N), R = R)
    # for i in eachindex(dyad_keys)
    #     pinv_sdv[dyad_keys[i]] = vec(pinv_matrix[:, i])           
    # end

    return dyad_dict
end


function dyad_run()
    N = 4
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)  
    add_channel_depolarizing!(L, .1)
    println("Lindbladian: ")
    display(L)

    Lmat = Matrix(L)
    dim_L = size(Lmat, 1)
    # println("Matrix Form of L: ")
    # println("Diagonalization started")

    F = eigen(Lmat)
    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]
    @printf(" Eigenvalues of L:\n")
    for i in (length(F.values) - 2):length(F.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states[i])))
    end

    display(size(Lmat))

    state = DyadSum(Dyad(N, 0, 0))
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)

    p_dyad, eig_sci = selected_ci(L, v0, max_iter_outer = 1)
    println("\n Eigenvalues after SCI")
    display(eig_sci)
    # lmat_pp = build_subspace_L(L, p_dyad)
    # println("P-Space")
    # display(lmat_pp)
    x_temp = L * p_dyad
    # println("P-Space")
    # display(p_dyad)

    p_projector = SparseDyadVectors(DyadSum(N), R = 1)
    proj_coeff = fill(1, 1)

    for (d, c) in p_dyad
        p_projector[d] = proj_coeff
    end

    x_dyad = SparseDyadVectors(DyadSum(N), R=nkeep)

    for (d, coeff) in x_temp
        if !haskey(p_dyad, d)
            sum!(x_dyad, d, coeff)
        end
    end 

    x_projector =SparseDyadVectors(DyadSum(N), R = 1)
    for (d, coeff) in x_temp
        x_projector[d] = proj_coeff
    end  

    # display(length(x_dyad))
    # display(p_dyad' * (L*p_dyad))

    # now getting the correction for the eigenvalues 
    c_idx = 1

    l_x_dyad = L * x_projector

    l_px = p_projector' * l_x_dyad 

    l_xp = x_projector' * (L * p_projector)
    l_xx = x_projector' * l_x_dyad

    # correction = l_px[c_idx, c_idx] * inv(eig_sci[c_idx] - l_xx[c_idx, c_idx]) * l_xp[c_idx, c_idx]
    correction = l_px[1, 1] * inv(eig_sci[c_idx] - l_xx[1, 1]) * l_xp[1, 1]

    println("\nTemp")

    # display(l_px)
    # a = left_eigenvectors(p_dyad, build_subspace_L(L, p_dyad))
    # display(inv(eig_sci[c_idx] - l_xx[1, 1]))

    println("\n Correction")
    display(correction)
    corr_eig = eig_sci[c_idx] + correction

    println("\n Corrected Eigenvalue")
    display(corr_eig)
    return
end

dyad_run()
# matrix_run()