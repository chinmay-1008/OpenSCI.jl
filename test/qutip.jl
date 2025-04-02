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

function run()
    N = 1
    dim = 2^N

    # Initializing the Lindbladian
    L = Lindbladian(N)
    # x_h = PauliSum(N) + 0.5*(Pauli(N, X = [1])) #+ Pauli(N, Y = [2]) + Pauli(N, Z = [3]))
    # add_hamiltonian!(L,x_h)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, .1)
    add_channel_depolarizing!(L, .1)
    println("Lindbladian: ")
    display(L)

    Lmat = Matrix(L)
    println("Matrix Form of L: ")
    display(Lmat)

    state = DyadSum(Dyad(N, 0, 0))

    # Getting the Eigenvalues of Lmat
    F = eigen(Lmat)

    # Sort Eigenvalues by real part
    # perm = sortperm(F.values, by=real)
    # F.values .= F.values[perm]
    # F.vectors .= F.vectors[:, perm]
    T = 10.0
    vec_state = vec(Matrix(state))
    states = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]
    @printf(" Eigenvalues of L:\n")
    for i in 1:length(F.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states[i])))
    end
   

    # Computing the ρt using the Eigenvalues
    ρt = compute_ρt(T, F, vec_state)
    ρt = reshape(ρt, (dim, dim))
    println("\nUsing Eigenvalues: ")
    display(ρt)
    println("Expectation Value")
    display(tr(Matrix(Pauli(N, Y = [1]))*ρt))

    f(du, u, p, t) = du .= Lmat*u
    tspan = (0.0, T)
    prob = ODEProblem(f, vec_state, tspan) 
    sol = solve(prob, reltol = 1e-6, abstol = 1e-8)

    ρT = reshape(sol.u[end], (2^N, 2^N))
    println("\nUsing ODE:")
    display(ρT)
    println("Expectation Value")
    display(tr(Matrix(Pauli(N, Y = [1]))*ρT))

    # Using the SCI formalism
    v0 = DyadSum(Dyad(N,0,0))

    v0 = SparseDyadVectors(v0)
    dim_1, R = size(v0)

    final_state = selected_ci(L, v0, max_iter_outer=10)
    display(final_state)
    Lmat_sci = build_subspace_L(L, final_state)
    display(Lmat_sci)
    F_sci = eigen(Lmat_sci)

    perm = sortperm(F_sci.values, by=real)
    F_sci.values .= F_sci.values[perm]
    F_sci.vectors .= F_sci.vectors[:, perm]
    final_vec_state = vec(Matrix(final_state))
    # display(F_sci.values)

    # Computing the ρt using the Eigenvalues
    ρt = compute_ρt(T, F_sci, final_vec_state)
    println(size(Matrix(Pauli(N, Y = [1]))), " ", size(ρt))

    # ρt = reshape(ρt, (dim, dim))
    println("\nUsing Eigenvalues SCI: ")
    display(ρt)
    # println("Expectation Value")
    # display(tr(Matrix(Pauli(N, Y = [1]))*ρt))


    # println("\nExpectation Value: ")
    # exp_val = tr(Matrix(Pauli("Y"))*ρT)
    # println(exp_val)

    # For plotting the population during te dynamics
    # populations_ode = Dict{Dyad{N}, Vector{Float64}}([])
    # populations_eig = Dict{Dyad{N}, Vector{Float64}}([])
    # for i in 0:2^N-1
    #     ii = Dyad(N,i,i)
    #     ii_idx = index(ii)
    #     populations_ode[ii] = [abs(v[ii_idx]) for v in sol.u]
    #     populations_eig[ii] = []
    # end

    # for ti in 1:length(sol.t)
    #     t = sol.t[ti]
    #     ρt = compute_ρt(t, F, vec_state)
    #     ρode = sol.u[ti]
    #     @test norm(ρt - ρode) < 1e-6
    #     for i in 0:2^N-1
    #         ii = Dyad(N,i,i)
    #         ii_idx = index(ii)
    #         push!(populations_eig[ii], abs(ρt[ii_idx]))
    #     end
    # end
    # t = [i for i in sol.t]
    # plt = plot()
    # for ei in λ 
    #     # push!(t, [abs(exp(ei*ti)) for ti in t], linestyle = :dash, c=:gray)
    #     plot!(t, [abs(exp(ei*ti)) for ti in t], linestyle = :dash, c=:gray)
    # end
    # for (state, pops) in populations_ode
    #     plot!(t, pops, label = string(state.bra.v+1), c = palette(:tab10)[state.bra.v+1])
    # end
    # for (state, pops) in populations_eig
    #     plot!(t, pops, label = string(state.bra.v+1), linestyle=:dash, c = palette(:tab10)[state.bra.v+1])
    # end
    # savefig("./plot.pdf")

    return
end

run()