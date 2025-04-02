using PauliOperators
using OpenSCI
using Test
using LinearAlgebra
using Printf
using Random
using Plots
using DifferentialEquations
using KrylovKit
using Arpack

# @testset "tests1.jl" begin
function test1()
    Random.seed!(1234)
    N = 2

    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, .1)
    add_channel_depolarizing!(L, .1)
    # add_channel_amplitude_damping!(L, .1)
    display(L)

    # Lmat = Matrix(L)
    # @show norm(Lmat*Lmat' - Lmat'*Lmat)
    # :LR: eigenvalues with largest (most positive) real part
    

    # vals, vecs, info = eigsolve(Matrix(L), 6, :LR,  
    #     verbosity=1, tol=1e-14, maxiter=1000, 
    #     ishermitian=false, issymmetric=false
    # )
    # vals, vecs = eigs(Matrix(L); nev=100, which=:LR, tol=1e-9, maxiter=1000) 

    vals, vecs = eigen(Matrix(L))

    println(" v'v: ")
    display(vecs[:,end]'*vecs[:,end])
    states = [reshape(vecs[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(vals)]
    # states = [reshape(vecs[:,i], 2^N, 2^N) for i in 1:length(vals)]

    # for i in states
    #     display(i)
    # end
    
    @printf(" Eigenvalues of L:\n")
    for i in 1:length(vals)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(vals[i]), imag(vals[i]), real(tr(states[i])))
    end
   
    
    v0 = DyadSum(Dyad(N,0,0))
    @test abs(tr(reshape(Matrix(v0 + L*v0), 2^N, 2^N)) - 1) < 1e-14

    v0 = SparseDyadVectors(v0)
    display(v0)
    display(Vector(v0))
    display(Matrix(v0))
    

    tmp1 = SparseDyadVectors{N,ComplexF64}()
    for i in 1:20
        tmp1[rand(DyadBasis{N})] = [randn() + 1im*randn() for j in 1:4]
    end
    @test norm(Matrix(tmp1) - Matrix(fill!(tmp1,Matrix(tmp1)))) < 1e-14

    println()
    # display(L*v0)

    err = Matrix(L)*todense(v0) - todense(L*v0)
    @test isapprox(norm(err), 0, atol=1e-14)

    final_state = selected_ci(L, v0, max_iter_outer=5)

    Lmat = Matrix(L)
    l = eigvals(Lmat)
    @printf("\n Eigenvalues of Lmat:\n")
    for i in 1:length(l)
        @printf(" %4i % 12.8f % 12.8fi\n", i, real(l[i]), imag(l[i]))
    end

    display(reshape(todense(final_state), 2^N, 2^N))
end

test1()