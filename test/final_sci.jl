using OpenSCI
using PauliOperators
using LinearAlgebra
# using DifferentialEquations
using Plots, Measures
using Printf
using OrderedCollections
using StatProfilerHTML
# using BenchmarkTools
# using Arpack
gr(display_type=:inline)


function pinv_sparsedyads(dyad_dict::SparseDyadVectors{N,T})::SparseDyadVectors{N,T} where {N,T}
    dyad_keys = collect(keys(dyad_dict))                            
    values_matrix = transpose(hcat(values(dyad_dict)...))           

    pinv_matrix = pinv(values_matrix)                              

    pinv_sdv = OrderedDict{DyadBasis{N}, Vector{T}}()
    for i in eachindex(dyad_keys)
        pinv_sdv[dyad_keys[i]] = vec(pinv_matrix[:, i])           
    end

    return pinv_sdv
end

function run_sci(L:: Lindbladian{N}; sci_iter = 2) where {N}
    
    state = DyadSum(Dyad(N, 0, 0))
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)

    p_dyad, eig_sci = selected_ci(L, v0, ϵdiscard=1e-4, max_iter_outer = sci_iter)

    return p_dyad, eig_sci
end


function run_sci_first_order(L:: Lindbladian{N}; sci_iter = 2) where {N}
    
    state = DyadSum(Dyad(N, 0, 0))
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)

    p_dyad, eig_sci = selected_ci_first_correction(L, v0, ϵdiscard=1e-4, max_iter_outer = sci_iter)

    return p_dyad, eig_sci
end


function run_dyad_form(L::Lindbladian{N}, p_dyad::SparseDyadVectors{N,T}, eig_sci :: Union{ComplexF64, Float64}, corr_idx :: Int64) where {N, T}
    dim, nkeep = size(p_dyad)
    # Get the component of L*|v_i>> in the external space Q
    x_temp = L * p_dyad

    x_dyad = SparseDyadVectors(DyadSum(N), R=nkeep)
    # This represents |q>><<q|L|v_i>>
    for (d, coeff) in x_temp
        if !haskey(p_dyad, d)
            sum!(x_dyad, d, coeff)
        end
    end

    # Get the left eigenvectors <<W_i| of the projected Lindbladian P*L*P
    # left_p_dyad = left_eigenvectors(p_dyad, build_subspace_L(L, p_dyad))
    left_p_dyad = pinv_sparsedyads(p_dyad)

    correction_vec = zeros(ComplexF64, nkeep)
    println("\n Q space size: ", size(x_dyad))

    for (q_dyad, coeff) in x_dyad
        
        # Applying L to |q>> and projecting onto the left eigenvectors <<W_i|.
        q_vec_scalar = SparseDyadVectors(DyadSum(q_dyad), R = 1)
        L_q_scalar = L * q_vec_scalar 
        
        num_1 = zeros(ComplexF64, nkeep)
        num_1 = vec(num_1)

        for (p_k, d_k_vec) in left_p_dyad
            Lq_pk_coeff = get(L_q_scalar, p_k, 0.0 + 0im)
            num_1 .+= (d_k_vec) .* Lq_pk_coeff
        end

        # Calculate the denominator: λ_i - <<x|L|x>>
        q_vec_vector = SparseDyadVectors(DyadSum(q_dyad), R = nkeep)
        q_vec_vector[q_dyad] = [1, 1]
        L_q_vector = L * q_vec_vector 
        
        denominator_diag_vec = get(L_q_vector, q_dyad, vec(zeros(ComplexF64, nkeep)))
        threshold = 1e-10
        
        # Replace small values with threshold (element-wise)
        denominator_diag_vec .= ifelse.(abs.(denominator_diag_vec) .< threshold, threshold, denominator_diag_vec)
        energy_diff_vec = eig_sci .- denominator_diag_vec

        term_vec = (num_1 .* coeff) ./ energy_diff_vec
        
        correction_vec .+= term_vec
    end

    correction = correction_vec[corr_idx]
    
    corr_eig = eig_sci + correction

    return corr_eig
end


function run_matrix_form(L::Lindbladian{N}, p_dyad::SparseDyadVectors{N,T}, eig_sci :: Union{ComplexF64, Float64}, corr_idx :: Int64) where {N, T}

    dim, nkeep = size(p_dyad)
    vec_p_dyad = Matrix(p_dyad)

    lmat_pp = build_subspace_L(L, p_dyad)

    x_temp = L*p_dyad

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

    lambda = eig_sci*identity_matrix

    lmat_xx = Diagonal(diag(lmat_xx))

    pertubation  = lmat_px * inv(lambda - lmat_xx) * lmat_xp
    leff = lmat_pp + pertubation

    # e, v = eigen(leff)
    # println("\n Eigenvalues after Lowdin")
    # new_eig = e[end-nkeep+1:end]
    # display(new_eig)

    println("\n Using SCI eigenvector")
    # Getting the left eigenvectors
    # left_eigen = left_eigenvectors(p_dyad, lmat_pp)
    # vec_left = Matrix(left_eigen)

    left_eigen = pinv_sparsedyads(p_dyad)
    vec_left = Matrix(left_eigen)
    # display(transpose(vec_left[:, corr_idx]) * vec_p_dyad[:, corr_idx])
    # sci_vec = (vec_left[:, corr_idx])' * leff * vec_p_dyad[:, corr_idx] 

    sci_vec = transpose(vec_left[:, corr_idx]) * leff * vec_p_dyad[:, corr_idx] # not a conjugate here
    # display(sci_vec)

    return sci_vec
end



function run()
    N = 6
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

    sci_iters = 1:9
    corr_idx = 1
    exact_eig = F.values[end - (2 - corr_idx)]  
    
    sci_errors = Float64[]
    sci_errors_first_order = Float64[]

    matrix_errors = Float64[]

    dyad_errors = Float64[]
    dyad_errors_first_order = Float64[]

    subspace_sizes          = Int[]  # for SCI + Löwdin
    subspace_sizes_first    = Int[]  # for first-order SCI + Löwdin
    
    for iter in sci_iters
        println("\n SCI ITERATION CAP: ", iter)
        p_dyad, eig_sci = run_sci(L, sci_iter = iter)
        p_dyad_first_order, eig_sci_first_order = run_sci_first_order(L, sci_iter = iter)

        push!(subspace_sizes, length(p_dyad))
        push!(subspace_sizes_first, length(p_dyad_first_order))
    
        lambda_sci = eig_sci[corr_idx]

        lambda_sci_first_order = eig_sci_first_order[corr_idx]
    
        lambda_matrix = run_matrix_form(L, p_dyad, lambda_sci, corr_idx)
        
        lambda_dyad = run_dyad_form(L, p_dyad, lambda_sci, corr_idx)

        lambda_dyad_first_order = run_dyad_form(L, p_dyad_first_order, lambda_sci_first_order, corr_idx)
        

        # push!(sci_errors, abs(real(lambda_sci) - real(exact_eig)))
        # push!(matrix_errors, abs(real(lambda_matrix) - real(exact_eig)))
        # push!(dyad_errors, abs(real(lambda_dyad) - real(exact_eig)))
  
        push!(sci_errors, abs((lambda_sci) - (exact_eig)))
        push!(sci_errors_first_order, abs(lambda_sci_first_order - exact_eig))
        push!(matrix_errors, abs((lambda_matrix) - (exact_eig)))
        push!(dyad_errors, abs((lambda_dyad) - (exact_eig)))
        push!(dyad_errors_first_order, abs((lambda_dyad_first_order) - (exact_eig)))

        # push!(sci_errors, real(lambda_sci))
        # push!(matrix_errors, real(lambda_matrix))
        # push!(dyad_errors,real(lambda_dyad) )

        println("\n Error in Eigenvalue SCI ", length(p_dyad))
        display(lambda_sci - exact_eig)

        println("\n Error in Eigenvalue SCI with first order ", length(p_dyad_first_order))
        display(lambda_sci_first_order - exact_eig)

        # println("\n Error in Eigenvalue after Lowdin (MATRIX)")
        # display(matrix_errors)

        println("\n Error in Eigenvalue after Lowdin (DYAD)")
        display(lambda_dyad - exact_eig)

        println("\n Error in Eigenvalue after Lowdin (DYAD) with first order")
        display(lambda_dyad_first_order - exact_eig)

    end
    # Plot
    lambda_idx = 2 - corr_idx

    plot(
        subspace_sizes, sci_errors;
        label = "E0",
        lw = 1.5,
        ms = 3,
        marker = :circle,
        yscale = :log10,
        xlabel = "SCI Subspace Size",
        ylabel = "Absolute Error",
        title = "Error vs Subspace Size of λ_$lambda_idx for N = $N",
        grid = true,
        dpi = 300,
        color = :blue,     
    )
    
    plot!(subspace_sizes_first, sci_errors_first_order;
        label = "E0 - First Order",
        lw = 1.5,
        ms = 3,
        marker = :circle,
        linestyle = :dashdotdot,
        color = :blue,  
    )
    
    plot!(subspace_sizes, dyad_errors;
        label = "E2",
        lw = 1.5,
        ms = 3,
        marker = :diamond,
        color = :red,       
    )
    
    plot!(subspace_sizes_first, dyad_errors_first_order;
        label = "E2 - First Order",
        lw = 1.5,
        ms = 3,
        marker = :diamond,
        linestyle = :dashdotdot, 
        color = :red,   
    )

#     plot!(subspace_sizes, matrix_errors;
#     label = "Matrix Formalism",
#     lw = 1.5,
#     ms = 3,
#     marker = :utriangle,
#     linestyle = :dashdot, 
#     # color = :red,   
# )
    


    savefig("test/corr_lowdin_$N-matrix-$corr_idx.png")

end    

# σxx = (L * Pv)/(Lxx - λ) 

function run_first_corr()
    N = 6
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

    iter = 10

    p_dyad, eig_sci = run_sci(L, sci_iter = iter)

    p_dyad_first, eig_sci_first = run_sci_first_order(L, sci_iter = iter)
    println("NORMAL SCI")
    display(eig_sci)

    println("FIRST ORDER CORRECTION")
    display(eig_sci_first)
    return

end

# run_first_corr()
run()