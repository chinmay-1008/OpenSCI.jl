using PauliOperators
using OpenSCI
using Test
using LinearAlgebra
using Printf
using Random

@testset "Pauli Evolve" begin
    Random.seed!(1234)
    N = 2
    T = ComplexF64 
    for i in 1:10
        p = rand(PauliSum{N,T}, n_paulis=10)
        g = Pauli(rand(PauliBasis{N}))
        pm = Matrix(p)
        gm = Matrix(g)

        err = exp(1im * gm/2) * pm * exp(-1im * gm/2) - Matrix(OpenSCI.evolve(p,g))
        @test norm(err) < 1e-12
    end
end