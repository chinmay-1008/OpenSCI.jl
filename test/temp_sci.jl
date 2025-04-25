using OpenSCI
using PauliOperators
using LinearAlgebra
using DifferentialEquations
using Plots, Measures
using Printf
using OrderedCollections
using StatProfilerHTML
gr(display_type=:inline)


function compute_ρt_exp(t, F, ρ0::Vector, mat_ops, dim)
        
    w = inv(F.vectors)
    v = F.vectors
    λ = F.values
    ρt = v * Diagonal(exp.(λ*t)) * w * ρ0
    ρt = reshape(ρt, (dim, dim))
    exp_eig = tr(mat_ops*ρt)
    popu = []
    for i in 1:dim
        push!(popu, ρt[i, i])
    end
    # display(ρt)
    return exp_eig, popu
end

function compute_ρt_ss_exp(t, Fe, Fv, ρ0::Vector, mat_ops, dim)
        
    w = pinv(Fv)
    v = Fv
    λ = Fe
    ρtss = v * Diagonal(exp.(λ*t)) * w * ρ0
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
        ovi = sum(conj(d_ops[d]) * vi[d][m] for d in keys(vi) if haskey(d_ops, d))
        wir = sum((wi[d][m]) * v0[d][1] for d in keys(wi) if haskey(v0, d)) # Should it be conj(wi)?
        expval += exp(ei[m]*t) * ovi * wir
    end
    return expval
end

function population_densities(N, vi, wi, v0, ei, t, R)
    pops = zeros(2^N)
    for i in 0:2^N-1
        obs = DyadSum(Dyad(N, i, i))
        pops[i+1] = abs(expectation_sparse(vi, wi, obs, v0, ei, t, R))
    end
    return pops
end


function run()
    N = 4
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 10.1, 1.2, 5.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, .1)
    println("Lindbladian: ")
    display(L)

    Lmat = Matrix(L)
    # println("Matrix Form of L: ")
    # display(Lmat)
    println("Diagonalization started")
    F = eigen(Lmat)

    state = DyadSum(Dyad(N, dim-1, dim-1))
    vec_state_i = vec(Matrix(state))

    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]
    @printf(" Eigenvalues of L:\n")
    for i in 1:length(F.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states[i])))
    end

    sci_val = []
    eig_val = []
    eig_val_ss = []
    pop_t_sci = []
    pop_t_ex = []
    ops = Pauli(N, Z = [1, 2])
    # ops += Pauli(N, Z = [2,3])
    mat_ops = Matrix(ops)
    d_ops = matrix_to_dyad(mat_ops)

    # Number of Eigenvectors for SCI
    nkeep = 5

    v0 = SparseDyadVectors(state, R = nkeep)
    println("SCI started")
    final_state, ei = selected_ci(L, v0, max_iter_outer=10)
    # Lmat_sci = build_subspace_L(L, final_state)

    # F_sci = eigen(Lmat_sci)

    # perm = sortperm(F_sci.values, by=real)
    # F_sci.values .= F_sci.values[perm]
    # F_sci.vectors .= F_sci.vectors[:, perm]
    # ei = F_sci.values
    # vi_f = F_sci.vectors
    # ei = ei[end-R_1+1:end]
    # vi_f = vi_f[:, end-R_1+1:end]
    
    dim_1, R_1 = size(final_state)
    # println(dim_1, " ", R_1)

    
    states_sci = [reshape(Matrix(todense(final_state))[:, i], 2^N, 2^N)/sqrt(2^N) for i in 1:nkeep]
    @printf(" Eigenvalues of L SCI:\n")
    for i in 1:nkeep
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(ei[i]), imag(ei[i]), real(tr(states_sci[i])))
    end

    # Initializing the left and right eigenvectors
    vi = final_state
    wi = pinv_sparsedyads(vi)

    mat_vi = Matrix(todense(vi))

    time_step = [i/20 for i in 0:100]

    for T in time_step
        println("==========================EXP==================================")
        # Exact Formalism
        exp_eig, pop_ex = compute_ρt_exp(T, F, vec_state_i, mat_ops, dim)

        println("===========================EXPSS=================================")

        # SCI Dense Formalism
        exp_eigss = compute_ρt_ss_exp(T, ei, mat_vi, vec_state_i, mat_ops, dim)

        println("==========================EXPSPARSE==================================")

        # SCI Sparse Formalism
        out_n = expectation_sparse(vi, wi, d_ops, v0, ei, T, R_1)

        pop_sci = population_densities(N, vi, wi, v0, ei, T, R_1)

        println("Time: ", T)
        println("\n Exp Value using SCI Sparse")
        display(out_n)
        println("\n Exp Value using Eigen Values")
        display(exp_eig)
        println("\n Exp Value using SCI Dense")
        display(exp_eigss)
        display(abs(exp_eig) / abs(out_n))

        push!(sci_val, abs(out_n))
        push!(eig_val, abs(exp_eig))
        push!(eig_val_ss, abs(exp_eigss))
        push!(pop_t_sci, pop_sci)
        push!(pop_t_ex, pop_ex)
    end
    s_ops = string(ops)
    f_size = 8
    plot(time_step, [sci_val, eig_val, eig_val_ss], 
        label = ["SCI" "Eig" "Eig(ss)"],
        xlabel = "Time (t)", 
        ylabel = "Expectation value ⟨O⟩(t)",
        title = "Expectation value of $s_ops using SCI(R = $R_1) and Eigendecomposition of L for N=$N",
        legend = :topright,
        lw = 1,
        marker = :circle,
        guidefontsize = f_size,     
        tickfontsize = f_size,      
        legendfontsize = f_size,    
        titlefontsize = f_size) 
    savefig("test/sci_vs_eig_$N-r_$R_1.pdf")

    label_exact = reduce(hcat, [["Exact |$(i)⟩"] for i in 0:2^N-1])
    label_sci   = reduce(hcat, [["SCI   |$(i)⟩"] for i in 0:2^N-1])

    Y1 = abs.(hcat(pop_t_ex...)')  # Exact results
    Y2 = abs.(hcat(pop_t_sci...)')  # SCI results
    # display(Y2)
    p1 = plot(time_step, Y1,
        label=label_exact,
        # linestyle=:solid,
        title="Exact",
        xlabel="Time", ylabel="Population",
        legend=:topright
    )

    p2 = plot(time_step, Y2,
        label=label_sci,
        # linestyle=:dash,
        title="SCI",
        xlabel="Time", ylabel="Population",
        legend=:topright
    )


    plot(p1, p2, layout=(1, 2), size=(1000, 400), top_margin=5mm,bottom_margin = 5mm, right_margin=5mm, left_margin=5mm, dpi=300, legendfontsize =4)
    savefig("test/pop_$N.pdf")

    return
end

function decay_rate(N)
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 2.1, 1.2, 1.3))
    add_channel_dephasing!(L, .1)
    add_channel_depolarizing!(L, .1)
    # println("Lindbladian: ")
    # display(L)
    nkeep = 4

    Lmat = Matrix(L)
    # println("Matrix Form of L: ")
    # display(Lmat)
    println("Diagonalization started")
    F = eigen(Lmat)

    state = DyadSum(Dyad(N, dim-1, dim-1))
    vec_state_i = vec(Matrix(state))

    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]
    @printf(" Eigenvalues of L:\n")
    num_eig_ex = length(F.values)
    for i in num_eig_ex-nkeep:num_eig_ex
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states[i])))
    end

    # Number of Eigenvectors for SCI

    v0 = SparseDyadVectors(state, R = nkeep)
    println("SCI started")
    final_state, ei = selected_ci(L, v0, max_iter_outer=10)

    dim_1, R_1 = size(final_state)
    # println(dim_1, " ", R_1)

    
    states_sci = [reshape(Matrix(todense(final_state))[:, i], 2^N, 2^N)/sqrt(2^N) for i in 1:nkeep]
    @printf(" Eigenvalues of L SCI:\n")
    for i in 1:nkeep
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(ei[i]), imag(ei[i]), real(tr(states_sci[i])))
    end

end

# run()
for i in 2:8
    println("N: ", i)
    decay_rate(i)
end
