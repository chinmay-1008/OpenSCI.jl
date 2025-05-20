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


function run()
    N = 2
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
    for i in 1:length(F.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states[i])))
    end

    display(size(Lmat))

    state = DyadSum(Dyad(N, 0, 0))
    vec_state_i = vec(Matrix(state))
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)

    state_sci, eig_sci = selected_ci(L, v0, max_iter_outer = 10)

    v_p = Matrix(todense(state_sci))
    
    D, R = size(v_p)
    
    P_space = v_p * v_p'
    println("P-Space")
    display(state_sci)

    Q_space = Matrix{eltype(v_p)}(I, D, D) - P_space
    v_q = qr(Q_space).Q[:, R+1:end]      

    q_temp = L*state_sci
    # println("Q-Space")
    # display(q_temp)

    q_dyad = SparseDyadVectors(DyadSum(N), R=nkeep)

    for (d, coeff) in q_temp
        if !haskey(state_sci, d)
            sum!(q_dyad, d, coeff)
        end
    end 

    println("Just Q")
    display(q_dyad)

    l_qq = L * q_dyad
    lqq_action = SparseDyadVectors(DyadSum(N), R=nkeep)
 
    for (d, coeff) in l_qq
        if haskey(q_dyad, d)
            sum!(lqq_action, d, coeff)
        end
    end 
    println("L*Q")
    display(lqq_action)

    L_PP = v_p' * Lmat * v_p
    L_PQ = v_p' * Lmat * v_q
    L_QP = v_q' * Lmat * v_p
    L_QQ = v_q' * Lmat * v_q
    # display(Lmat * v_p[:, 2])
    # count = 0
    ω = eig_sci[1]
    # # for ω in eig_sci
    L_eff = L_PP + L_PQ * inv(ω * I - L_QQ) * L_QP
    # display(L_eff)
    e, v = eigen(L_eff)
    display(ω)
    println("================")
    # display(e)
    display(size(L_PQ * inv(ω * I - L_QQ) * L_QP))

    # display(ω + v_p[:, 1]' * L_PQ * inv(ω * I - L_QQ) * L_QP * v_p[:, 1])
    # println("+++++++++++++++++++++++")
    # ρ_P = v_p' * vec_state_i
    # # display(ρ_P)
    # time_step = [i/10 for i in 0:50]
    # plot_population_lowdin_eigen(e, v, v_p, ρ_P, time_step, count )
    # count+=1
    # end

    return 
end

run()