using OpenSCI
using PauliOperators
using LinearAlgebra
using DifferentialEquations
using Plots
using Printf
using OrderedCollections


function compute_ρt_exp(t, F, ρ0::Vector, mat_ops, dim)
        
    w = inv(F.vectors)
    v = F.vectors
    λ = F.values
    ρt = v * Diagonal(exp.(λ*t)) * w * ρ0
    ρt = reshape(ρt, (dim, dim))
    exp_eig = tr(mat_ops*ρt)

    return exp_eig
end

function compute_ρt_ss_exp(t, Fe, Fv, ρ0::Vector, mat_ops, dim)
        
    w = pinv(Fv)
    v = Fv
    λ = Fe
    ρt_ss = v * Diagonal(exp.(λ*t)) * w * ρ0
    ρtss = reshape(ρtss, (dim, dim))
    exp_eigss = tr(mat_ops*ρtss)
    return exp_eigss
end

function matrix_to_dyad(mat_p)
    n, m = size(mat_p)
    N = Int(log2(n))
    dyad_pauli = DyadSum(N)
    for i in 1:m
        for j in 1:n
            temp = mat_p[i, j]
            
            if abs(temp) ≠ 0
                dyad_pauli += temp*Dyad(N, i-1, j-1)
            end
        end
    end  
    return dyad_pauli  
end

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

function expectation_sparse(vi, wi, d_ops, v0, ei, t, R)
    expval = 0.0
    for m in 1:R
        ovi = sum(conj(d_ops[d]) * vi[d][m] for d in keys(vi) if haskey(d_ops,d))
        wir = sum(conj(wi[d][m]) * v0[d][1] for d in keys(wi) if haskey(v0,d))
        expval += exp(ei[m]*t) * ovi * wir
    end
    return expval
end


function run()
    N = 6
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, .6)
    add_channel_depolarizing!(L, .1)
    println("Lindbladian: ")
    display(L)

    Lmat = Matrix(L)
    # println("Matrix Form of L: ")
    # display(Lmat)
    println("Diagonalization started")
    @time F = eigen(Lmat)

    state = DyadSum(Dyad(N, dim-1, dim-1))
    vec_state_i = vec(Matrix(state))

    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]
    # @printf(" Eigenvalues of L:\n")
    # for i in 1:length(F.values)
    #     @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states[i])))
    # end

    sci_val = []
    eig_val = []
    eig_val_ss = []

    ops = Pauli(N, Z = [1, 2])
    # ops += Pauli(N, Z = [2,3])
    mat_ops = Matrix(ops)
    d_ops = matrix_to_dyad(mat_ops)

    # Number of Eigenvectors for SCI
    nkeep = 3

    v0 = SparseDyadVectors(state, R = nkeep)
    println("SCI started")
    @time final_state = selected_ci(L, v0, max_iter_outer=10)

    Lmat_sci = build_subspace_L(L, final_state)

    F_sci = eigen(Lmat_sci)

    perm = sortperm(F_sci.values, by=real)
    F_sci.values .= F_sci.values[perm]
    F_sci.vectors .= F_sci.vectors[:, perm]

    dim_1, R_1 = size(final_state)
    # println(dim_1, " ", R_1)

    ei = F_sci.values
    vi_f = F_sci.vectors
    ei = ei[end-R_1+1:end]
    vi_f = vi_f[:, end-R_1+1:end]
    
    states_sci = [reshape(Matrix(todense(final_state))[:, i], 2^N, 2^N)/sqrt(2^N) for i in 1:nkeep]
    # @printf(" Eigenvalues of L SCI:\n")
    # for i in 1:nkeep
    #     @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(ei[i]), imag(ei[i]), real(tr(states_sci[i])))
    # end

    # Initializing the left and right eigenvectors
    vi = final_state
    wi = pinv_sparsedyads(vi)

    mat_vi = Matrix(todense(vi))

    time_step = [i for i in 1:10]

    for T in time_step

        # Exact Formalism
        @time exp_eig = compute_ρt_exp(T, F, vec_state_i, mat_ops, dim)

        println("============================================================")

        # SCI Dense Formalism
        @time exp_eigss = compute_ρt_ss_exp(T, ei, mat_vi, vec_state_i, mat_ops, dim)

        println("============================================================")

        # SCI Sparse Formalism
        @time out_n = expectation_sparse(vi, wi, d_ops, v0, ei, T, R_1)

        # for m in 1:R_1
        #     for (state_v, coeff_v) in vi
        #         if haskey(d_ops, state_v)
        #             ovi = (d_ops[state_v])' * coeff_v[m]
        #             # println("OVI ")
        #             # display(ovi)
        #         else
        #             ovi = 0
        #         end
        #         # println("OVI")
        #         # display(ovi)

        #         if haskey(v0, state_v)
        #             wir = (wi[state_v][m]) * v0[state_v][m]
        #             # println("WIR")
        #             # display(wir) 

        #         else
        #             wir = 0
        #         end
        #         # println("WIR")
        #         # display(wir)
        #         out_n += (ovi * wir * exp(ei[m]*T))
        #         # println("OUT ")
        #         # display(out_n)
        #     end 
        # end 
        println("Time: ", T)
        println("\n Exp Value using SCI")
        display(out_n)
        println("\n Exp Value using Eigen Values")
        display(exp_eig)
        println("\n Exp Value using SCI Dense")
        display(exp_eigss)
        display(abs(exp_eig) / abs(out_n))

        push!(sci_val, abs(out_n))
        push!(eig_val, abs(exp_eig))
        push!(eig_val_ss, abs(exp_eigss))
    end
    s_ops = string(ops)
    f_size = 8
    plot(time_step, [sci_val, eig_val, eig_val_ss], 
        label = ["SCI" "Eig" "Eig(ss)"],
        xlabel = "Time (t)", 
        ylabel = "Expectation value ⟨O⟩(t)",
        title = "Expectation value of $s_ops using SCI(R = $R_1) and Eigendecomposition of L for N=$N",
        legend = :topright,
        lw = 2,
        marker = :circle,
        guidefontsize = f_size,     
        tickfontsize = f_size,      
        legendfontsize = f_size,    
        titlefontsize = f_size) 
    savefig("test/sci_vs_eig_$N-r_$R_1.pdf")
    return
end

run()