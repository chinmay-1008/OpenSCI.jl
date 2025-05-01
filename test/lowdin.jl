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
    println("Matrix Form of L: ")
    dim_1, dim_2 = size(Lmat)
    for i in 1:dim_1
        for j in i: dim_2
            println(i, " ", j, " ", Lmat[:, i]' * Lmat[:, j])
        end
    end
    # println("Diagonalization started")

    state = DyadSum(Dyad(N, 0, 0))
    # vec_state_i = vec(Matrix(state))
    v0 = SparseDyadVectors(state, R = 1)
    display(todense(v0))

    return 

end

run()