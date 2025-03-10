using PauliOperators
using OpenSCI
using Test
using LinearAlgebra
using Printf
using Random
using Plots
using Arpack
using LinearMaps
using BenchmarkTools

@testset "Lindbladian mul!" begin
    Random.seed!(1234)
    N = 6

    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, .1)
    add_channel_depolarizing!(L, .1)
    add_channel_amplitude_damping!(L, .1)
    display(L)

    v0 = rand(DyadSum{N}, n_terms=100)
    v0 = v0 * (1/tr(v0))

    v = zeros(ComplexF64, 4^N)
    for (d,c) in v0
        v[index(d)] = c
    end
    v = v / norm(v)

    Lmat = Matrix(L)
    σref = Lmat * v 

    σtst = zeros(ComplexF64,size(σref)...)
    
    mul!(σtst, L, v)

    @show norm(v)
    @show norm(σtst)
    @test norm(σtst - σref) < 1e-12
    
    @test norm(mul!(zero(similar(v)), L, v) - σref) < 1e-12
    @test norm(mul!(zero(similar(v)), L, v, 1, 0) - σref) < 1e-12

    @time mul!(zero(similar(v)), L, v)
    @time mul!(zero(similar(v)), Lmat, v)
   
    scr = zero(similar(v))
    # @show @allocated  mul!(scr, L, v)
    @printf(" Time Matrix mul!\n")
    @btime mul!($scr, $Lmat, $v)
    @printf(" Time Matrix free mul!\n")
    @btime mul!($scr, $L, $v)
    @printf(" Time Matrix free mul!\n")
    @btime mul!($scr, $L, $v, true, false)
   
    return
    # e,v = eigs(L; nev=5, which=:SM, tol=1e-6)
    Base.eltype(L::Lindbladian) = ComplexF64
  
    scr = zeros(ComplexF64, size(v)...)
    function mymatvec(v) 
        fill!(scr, 0)
        return mul!(scr, L, v)
    end

    Lmap = LinearMap{ComplexF64}(mymatvec, 4^N, 4^N)

    @test norm(Lmat*v - Lmap*v) < 1e-12
  
    @printf(" Now solve with Matrix mul!\n")
    eigvals1, eigvecs1 = eigs(Lmat, nev=5, which=:SM, tol=1e-5)
    for i in 1:length(eigvals1)
        @printf(" %3i %12.8f %12.8fi\n", i, real(eigvals1[i]), imag(eigvals1[i]))
    end
    
    @printf(" Now solve with Matrix-Free mul!\n")
    eigvals2, eigvecs2 = eigs(LinearMap(L), nev=5, which=:SM, tol=1e-5)
   
    for i in 1:length(eigvals2)
        @printf(" %3i %12.8f %12.8fi\n", i, real(eigvals2[i]), imag(eigvals2[i]))
    end
    
    # @test norm(eigvals1 - eigvals2) < 1e-12

end