using PauliOperators
using OpenSCI
using Test
using LinearAlgebra
using Printf
using Random
using Plots
using DifferentialEquations

# @testset "tests1.jl" begin
function run1()
    N = 2

    @test index(Dyad(6, 4, 8)) == 517 
    Λ = rand(Lindbladian{N}, nH=100, nL=0)
    display(Λ)

    ρ = DyadSum(Dyad(N,0,0)) 
    display(ρ)

    σ = DyadSum(N)
    
    display(typeof(ρ))
    display(typeof(σ))
    # rmul!(σ, Λ, ρ)
    σ = Λ*ρ

    # test this with Matrix
    ρv = Vector(ρ)
    Lv = Matrix(Λ)
    
    err = Matrix(Λ)*vec(Matrix(ρ)) - vec(Matrix(Λ*ρ))
    @test isapprox(norm(err), 0, atol=1e-14)
    
    err = -1im*(vec(Matrix(Λ.H)*Matrix(ρ) - Matrix(ρ)*Matrix(Λ.H))) - vec(Matrix(Λ*ρ))
    @test isapprox(norm(err), 0, atol=1e-14)

    U,s,V = svd(Lv)
    for i in 1:length(s)
        @printf(" %4i %12.8f %12.8fi\n", i, real(s[i]), imag(s[i]))
    end

   
    

    Λ = rand(Lindbladian{N}, nH=100, nL=0)

    display(Λ)
    F = eigen(Matrix(Λ.H))
    ρ = zeros(2^N, 2^N)
    for i in 1:length(F.values)
        ρ += rand()*F.vectors * F.vectors'
    end
    ρ = ρ/tr(ρ)
    # display(ρ)

    @show norm(Matrix(Λ)*vec(Matrix(ρ)))
   
    # return 
end

function run2()

    #### Now with non-unitary part 
    N = 1
    Λ = rand(Lindbladian{N}, nH=100, nL=0)
    display(Λ)
    @show ρ = DyadSum(Dyad(N,0,0)) 

    err = Matrix(Λ)*vec(Matrix(ρ)) - vec(Matrix(Λ*ρ))
    @test isapprox(norm(err), 0, atol=1e-14)

    U,s,V = svd(Matrix(Λ))
    
    println("\n Singular Values of Λ")
    for i in 1:length(s)
        @printf(" %4i %12.8f %12.8fi\n", i, real(s[i]), imag(s[i]))
    end
    # display(V[:,4])
    ρss = real(reshape(V[:,4], (2^N, 2^N)))
    ρss = ρss/tr(ρss)
    
    println("\n ρss")
    display(ρss)
    @printf(" tr(ρss) = %12.8f %12.8fi\n", real(tr(ρss)), imag(tr(ρss)))

    Random.seed!(2)
    ρ = Matrix{Float64}(I, 2^N, 2^N)
    
    F = svd(rand(2^N, 2^N))
    ρ = F.U * Diagonal(F.S) * F.U' / sum(F.S)

    println(" Eigenvalues of ρ ", eigvals(ρ))
    println("\n Trace of ρ: ", tr(ρ))
    println(" ρ")
    display(ρ)
    Λρ = Matrix(Λ)*vec(ρ)
    Λρ = reshape(Λρ, (2^N, 2^N))

    println(" Λρ")
    display(Λρ)
    println("\n Trace of ρ+Λρ: ", tr(ρ + Λρ))
    println(" Eigenvalues of ρ+Λρ ", eigvals(ρ + Λρ))

    println("\n  Do my jump operators sum to I?")
    tmp = PauliSum(N) 
    for Li in Λ.L
        tmp += Li*Li'
    end
    display(tmp)


    println("\n Let's try again")
    display(ρ)
    display(ρ + -1im*(Matrix(Λ.H)*ρ - ρ*Matrix(Λ.H)))
end

function heisenberg_test()
    N = 3
    dim = 2^N
    Λ = Lindbladian(N)
    # add_hamiltonian!(Λ, OpenSCI.heisenberg_1D(N, 1.0, 1.0, 1.0))
    # add_channel_dephasing!(Λ, .2)
    # add_channel_depolarizing!(Λ, .6)
    # add_channel_amplitude_damping!(Λ, .5)

    add_hamiltonian!(Λ, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(Λ, 0.1)
    add_channel_depolarizing!(Λ, .1)
    # Λ = rand(Lindbladian{N}, nH=10, nL=4)
    println(" Here is our Lindbladian:")
    display(Λ)
   

    Lmat = Matrix(Λ)*(1+0.0im)
   
    F = eigen(Lmat)

    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]

    V = F.vectors
    W = inv(F.vectors)
    λ = F.values
    # Make sure the diagonalization worked
    @test norm(Lmat - V*Diagonal(λ)*W) < 1e-12

    println("\n Eigenvalues of Λ")
    for i in 1:length(λ)
        vi = reshape(V[:,i], (dim, dim))
        # wi = reshape(W[:,i], (dim, dim))
        t = tr(vi)
        @printf(" %4i %12.8f %12.8fi Tr %12.8f %12.8fi\n", i, real(F.values[i]), imag(F.values[i]), real(t), imag(t))
    end

    
    ρ0 = DyadSum(Dyad(N,0,0))
    ρ0vec = vec(Matrix(ρ0))*(1+0.0im)
    println(" Initial State: ")
    display(ρ0)

    # This function will return the time dependent density Matrix
    function compute_ρt(t, F, ρ0::Vector)
        
        w = inv(F.vectors)
        v = F.vectors
        λ = F.values

        return v * Diagonal(exp.(λ*t)) * w * ρ0
    end
    
    T = 10 
    for i in [0.0, 0.1, 0.2, 0.5, 1.0] 
        ρt = compute_ρt(i, F, vec(Matrix(ρ0)))
        ρt = reshape(ρt, (dim, dim))
        evals = eigvals(ρt)
        # @show [real(i) for i in evals]        
        @test all([real(i) > -1e-14 && real(i) < 1+1e-14  for i in evals])
        @test all([abs(imag(i)) < 1e-14 for i in evals])
        # println(" tr(ρt) = ", tr(ρt))
    end


    f(du, u, p, t) = du .= Lmat*u
    tspan = (0.0, T)
    prob = ODEProblem(f, vec(Matrix(ρ0)), tspan) 
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8, saveat = 0.01)


    populations_ode = Dict{Dyad{N}, Vector{Float64}}([])
    populations_eig = Dict{Dyad{N}, Vector{Float64}}([])
    for i in 0:2^N-1
        ii = Dyad(N,i,i)
        ii_idx = index(ii)
        populations_ode[ii] = [abs(v[ii_idx]) for v in sol.u]
        populations_eig[ii] = []
    end
    for ti in 1:length(sol.t)
        t = sol.t[ti]
        ρt = compute_ρt(t, F, vec(Matrix(ρ0)))
        ρode = sol.u[ti]
        @test norm(ρt - ρode) < 1e-6
        for i in 0:2^N-1
            ii = Dyad(N,i,i)
            ii_idx = index(ii)
            push!(populations_eig[ii], abs(ρt[ii_idx]))
        end
    end
   
    t = [i for i in sol.t]
    plt = plot(dpi = 200)
    # for ei in λ 
    #     # push!(t, [abs(exp(ei*ti)) for ti in t], linestyle = :dash, c=:gray)
    #     plot!(t, [abs(exp(ei*ti)) for ti in t], linestyle = :dash, c=:gray)
    # end
    for (state, pops) in populations_ode
        plot!(t, pops, label = string(state.bra.v+1), c = palette(:tab10)[state.bra.v+1])
    end
    for (state, pops) in populations_eig
        plot!(t, pops, label = string(state.bra.v+1), linestyle=:dash, c = palette(:tab10)[state.bra.v+1])
    end
    savefig("./plot.png")
    
    return
    
    # t = [i for i in sol.t]
    # plot(t, gs_prob, label = "gs_prob")
    # return


    # gs_prob = [abs(i[1]*i[1]') for i in sol.u]
    # t = [i for i in sol.t]
    # plot(t, gs_prob, label = "gs_prob")
    # for ei in L_evals[1:5] 
    #     println(" ei = ", ei)
    #     plot!(t, [abs(exp(ei*ti)) for ti in t], linestyle = :dash, label = "exp(ei*ti)", legend=false)
    # end

    # for i in 1:length(sol.t)
    #     ρt = reshape(sol.u[i], (2^N, 2^N))
    #     @printf(" tr(ρt) = %12.8f %12.8fi\n", real(tr(ρt)), imag(tr(ρt)))
    # end
    # savefig("./plot.pdf")
    
    # ρT = reshape(sol.u[end], (2^N, 2^N))
    # println("\n ρT")
    # display(ρT)
    # println("\n eigvals(ρT)")
    # display(eigvals(ρT))
    # println("\n tr(ρT)")
    # display(tr(ρT))

    # println("\n tr(ρss*ρT)")
    # display(tr(ρss*ρT))
end

# run1()
# run2()
heisenberg_test()
