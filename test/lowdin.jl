using OpenSCI
using PauliOperators
using LinearAlgebra
using DifferentialEquations
using Plots, Measures
using Printf
using OrderedCollections
using StatProfilerHTML
using BenchmarkTools
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
    N = 4
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)  
    add_channel_depolarizing!(L, .1)
    # println("Lindbladian: ")
    # display(L)

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

    p_dyad, eig_sci = selected_ci(L, v0, max_iter_outer = 3)
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

    corr_idx = 1
    lambda = eig_sci[corr_idx]*identity_matrix

    pertubation  = lmat_px * inv(lambda - lmat_xx) * lmat_xp
    leff = lmat_pp + pertubation
    # println("\n Temp")
    # display(lmat_xx)

    # println("X-space")
    # display(x_dyad)
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
          
    e = 0
    v = zeros(T,size(dyad_dict))
    Lmat = Lmat'
    dim, R = size(dyad_dict)
    if length(dyad_dict) < 300
        e,v = eigen(Lmat)
        e = e[end-R+1:end]
        v = v[:, end-R+1:end]
    else
        e,v = eigs(Lmat, nev=R, v0=Matrix(Pv)[:,1], which=:LR, maxiter=3000 , tol=1e-5, check=1)
        perm = sortperm(real(e))
        e = e[perm]
        v = v[:, perm]
    end
    
    OpenSCI.fill!(dyad_dict, v)
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
    # println("Lindbladian: ")
    # display(L)

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

    p_dyad, eig_sci = selected_ci(L, v0, max_iter_outer = 4)
    println("\n Eigenvalues after SCI")
    display(eig_sci)


    # this is for Q * L * P |vi>>
    x_temp = L * p_dyad

    x_dyad = SparseDyadVectors(DyadSum(N), R=nkeep)

    for (d, coeff) in x_temp
        if !haskey(p_dyad, d)
            sum!(x_dyad, d, coeff)
        end
    end 

    # different approach

    # chi_coeffs = SparseDyadVectors(DyadSum(N), R=nkeep)

    # for (q_dyad, _) in x_dyad  

    #     γ = zeros(ComplexF64, 1, nkeep)
    #     γ = vec(γ)

    #     for (p_k, d_k) in p_dyad

    #         p_vec = SparseDyadVectors(DyadSum(p_k), R = 1)
    #         L_p = L * p_vec
    #         coeff = get(L_p, q_dyad, 0.0 + 0im)  
    #         γ .+= d_k .* coeff               
    #     end
    #     chi_coeffs[q_dyad] = γ
    # end

    
    # here we get the <<w_i| P * L * Q
    left_p_dyad = left_eigenvectors(p_dyad, build_subspace_L(L, p_dyad))

    gamma_coeffs = SparseDyadVectors(DyadSum(N), R=nkeep)

    for (q_dyad, _) in x_dyad  
        q_vec = SparseDyadVectors(DyadSum(q_dyad), R = 1)

        L_q = L * q_vec

        γ = zeros(ComplexF64, 1, nkeep)
        γ = vec(γ)
        for (p_k, d_k) in left_p_dyad
            coeff = get(L_q, p_k, 0.0 + 0im)  
            γ .+= conj.(d_k) .* coeff               
        end
        gamma_coeffs[q_dyad] = γ
    end
       
    xx_coeffs = SparseDyadVectors(DyadSum(N), R=nkeep)

    for (q_dyad, _) in x_dyad  
        q_vec = SparseDyadVectors(DyadSum(q_dyad), R = nkeep)

        L_q = L * q_vec

        coeff = get(L_q, q_dyad, 0.0 + 0im)         
        xx_coeffs[q_dyad] = coeff
    end

    correction = 0
    c_idx = 1

    for (q_dyad, coeff) in x_dyad
        correction += gamma_coeffs[q_dyad][c_idx] * coeff[c_idx] / (eig_sci[c_idx] - xx_coeffs[q_dyad][c_idx])
    end

    println("Correction")

    display(correction)
    corr_eig = eig_sci[c_idx] + correction

    println("\n Corrected Eigenvalue")
    display(corr_eig)

    return

end

using LinearAlgebra
using Printf
# Assuming OpenSCI, SparseDyadVectors, etc. are defined elsewhere
# include("path/to/OpenSCI_definitions.jl")

function dyad_run_optimized()
    N = 4
    dim = 2^N

    # Setup Lindbladian 
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, 0.1)

    Lmat = Matrix(L)
    dim_L = size(Lmat, 1)
    
    # Exact Diagonalization
    F = eigen(Lmat)
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]
    @printf(" Eigenvalues of L:\n")
    for i in (length(F.values) - 2):length(F.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states[i])))
    end
    display(size(Lmat))

    # Selected CI
    state = DyadSum(Dyad(N, 0, 0))
    nkeep = 2
    v0 = SparseDyadVectors(state, R=nkeep)
    p_dyad, eig_sci = selected_ci(L, v0, max_iter_outer=3)
    println("\n Eigenvalues after SCI")
    display(eig_sci)

    # Get the component of L*|V>> in the external space Q
    # This represents <<q|L|v_i>>
    x_temp = L * p_dyad
    x_dyad = SparseDyadVectors(DyadSum(N), R=nkeep)
    for (d, coeff) in x_temp
        if !haskey(p_dyad, d)
            sum!(x_dyad, d, coeff)
        end
    end

    # Get the left eigenvectors <<W_i| of the projected Lindbladian P*L*P
    left_p_dyad = left_eigenvectors(p_dyad, build_subspace_L(L, p_dyad))

    correction_vec = zeros(ComplexF64, nkeep)
    
    for (q_dyad, coeff) in x_dyad
        
        # Applying L to |q>> and projecting onto the left eigenvectors <<W_i|.
        q_vec_scalar = SparseDyadVectors(DyadSum(q_dyad), R = 1)
        L_q_scalar = L * q_vec_scalar 
        
        num_1 = zeros(ComplexF64, nkeep)
        num_1 = vec(num_1)
        for (p_k, d_k_vec) in left_p_dyad
            Lq_pk_coeff = get(L_q_scalar, p_k, 0.0 + 0im)
            num_1 .+= conj.(d_k_vec) .* Lq_pk_coeff
        end

        # Calculate the denominator: E_i - <<q|L|q>>
        q_vec_vector = SparseDyadVectors(DyadSum(q_dyad), R = nkeep)
        L_q_vector = L * q_vec_vector 
        
        denominator_diag_vec = get(L_q_vector, q_dyad, zeros(ComplexF64, nkeep))
        energy_diff_vec = eig_sci .- denominator_diag_vec

        term_vec = (num_1 .* coeff) ./ energy_diff_vec
        
        correction_vec .+= term_vec
    end

    c_idx = 1
    correction = correction_vec[c_idx]
    
    println("Correction")
    display(correction)
    
    corr_eig = eig_sci[c_idx] + correction
    println("\n Corrected Eigenvalue")
    display(corr_eig)

    return
end

# @btime dyad_run()
@time matrix_run()
# @time dyad_run_optimized()