using OpenSCI
using PauliOperators
using LinearAlgebra
using DifferentialEquations
using Plots, Measures
using Printf
using OrderedCollections
using StatProfilerHTML
gr(display_type=:inline)




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
    display(size(Lmat))
    state = DyadSum(Dyad(N, 0, 0))
    # vec_state_i = vec(Matrix(state))
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)

    state_sci, eig_sci = selected_ci(L, v0, max_iter_outer = 10)

    display(state_sci)
    display(build_subspace_L(L, state_sci))
    v_p = Matrix(todense(state_sci))
    P_space = v_p * v_p'
    display(P_space)

    i_mat = Matrix{ComplexF64}(LinearAlgebra.I, dim_L, dim_L)

    Q_space = i_mat - P_space
    display(Q_space)

    

    return 
end

run()