using OpenSCI
using PauliOperators
using LinearAlgebra
using DifferentialEquations
using Plots
using Printf

function compute_ρt(t, F, ρ0::Vector)
        
    w = inv(F.vectors)
    v = F.vectors
    λ = F.values

    return v * Diagonal(exp.(λ*t)) * w * ρ0
end

function compute_ρt_ss(t, Fe, Fv, ρ0::Vector)
        
    w = pinv(Fv)
    v = Fv
    λ = Fe
    println(size(w), " ", size(v), " ", size(λ))
    return v * Diagonal(exp.(λ*t)) * w * ρ0
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

function run()
    N = 4
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, .3)
    add_channel_depolarizing!(L, .1)
    println("Lindbladian: ")
    display(L)

    Lmat = Matrix(L)
    # println("Matrix Form of L: ")
    # display(Lmat)
    F = eigen(Lmat)

    state = DyadSum(Dyad(N, 0, 0))
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

    R_1 = 0
    # display(F.vectors[:, 7])
    ops = Pauli(N, Z = [1,2])
    ops += Pauli(N, Z = [2,3])

    # Using the SCI formalism

    v0 = DyadSum(Dyad(N,0,0))
    v0 = SparseDyadVectors(v0, R = 3)
    final_state = selected_ci(L, v0, max_iter_outer=10)

    # display(final_state)
    # display(todense(final_state))
    Lmat_sci = build_subspace_L(L, final_state)
    # display(Lmat_sci)

    F_sci = eigen(Lmat_sci)

    perm = sortperm(F_sci.values, by=real)
    F_sci.values .= F_sci.values[perm]
    F_sci.vectors .= F_sci.vectors[:, perm]

    # display(todense(v0))
    # println("\n State after SCI \n")
    # final_vec_state = vec(Matrix(final_state))
    # display(F_sci.values)
    # println("Eigenvalues of L subspace")
    # for i in 1:length(F_sci.values)
    #     @printf(" %4i %12.8f %12.8fi\n", i, real(F_sci.values[i]), imag(F_sci.values[i]))
    # end
    # return
    dim_1, R_1 = size(final_state)
    # println(dim_1, " ", R_1)

    ei = F_sci.values
    vi_f = F_sci.vectors
    ei = ei[end-R_1+1:end]
    vi_f = vi_f[:, end-R_1+1:end]
    # display(ei)
    # display(vi_f)
    # vi = todense(final_state)
    # wi = pinv(vi)
    vi = final_state
    wi = vi
    # println("Matrix form of State")
    # display(vi)

    # println("\n Inverse of State")
    # display(wi)
    # Here the pinv is same as the input so i am just taking the complex conjugate of the coeff

    # println("\n eigen decomposition")
    # display(vi * Diagonal(exp.(ei*T)) * wi)

   
    nkeep = 2 
    time_step = [i for i in 1:10]
    Fss_values = F.values[end-nkeep:end]
    Fss_vectors = F.vectors[:,end-nkeep:end]
    mat_vi = Matrix(todense(vi))
    display(size(mat_vi))
    display(ei)

    @show norm(mat_vi[:,end] - Fss_vectors[:,end])
    display(mat_vi' * Fss_vectors)

    for T in time_step

        # ρt = compute_ρt(T, F, vec_state_i)
        ρt = compute_ρt(T, F, vec_state_i)
        # ρt = compute_ρt(T, Matrix(todense(vi)), vec_state_i)
        ρt = reshape(ρt, (dim, dim))

        # @printf(" State after time T:\n")
        # display(ρt)
        # println("Expectation Value")
        exp_eig = tr(Matrix(ops)*ρt)
        # display(exp_eig)

        # display(F.vectors)

        ρtss = compute_ρt_ss(T, ei, mat_vi, vec_state_i)
        ρtss = reshape(ρtss, (dim, dim))
        exp_eigss = tr(Matrix(ops)*ρtss)
 
        println("============================================================")

        # display(ops*final_state)
        d_ops = matrix_to_dyad(Matrix(ops))
        out = 0
        # display(d_ops * Dyad(N, 1, 0))
        for m in 1:R_1
            for (state_v, coeff_v) in vi
                if haskey(d_ops, state_v)
                    ovi = d_ops[state_v]' * coeff_v[m]
                else
                    ovi = 0
                end
                # println("OVI")
                # display(ovi)

                if haskey(state, state_v)
                    wir = wi[state_v][m]' * state[state_v]
                else
                    wir = 0
                end
                # println("WIR")
                # display(wir)
                out += (ovi * wir * exp(ei[m]*T))
                # println("OUT ", out, " ", exp(ei[m]*T))
                # display(out)
            end 
        end 
        println("Time: ", T)
        println("\n Exp Value using SCI")
        display(out)
        println("\n Exp Value using Eigen Values")
        display(exp_eig)

        # Computing the ρt using the Eigenvalues
        # ρt = compute_ρt(T, F_sci, final_vec_state)
        # println(size(Matrix(Pauli(N, Y = [1]))), " ", size(ρt))

        push!(sci_val, abs(out))
        push!(eig_val, abs(exp_eig))
        push!(eig_val_ss, abs(exp_eigss))
    end

    plot(time_step, [sci_val,eig_val,eig_val_ss], label = ["SCI" "Eig" "Eig(ss)"])
    title!("Expectation value of Z_1 using SCI(R = $R_1) and Eigendecomposition of L for N=$N", titlefontsize = 8)
    savefig("test/sci_vs_eig_$N-r_$R_1.pdf")
    return
end

run()