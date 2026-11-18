using OpenSCI, PauliOperators, LinearAlgebra, SparseArrays, Random, Test
using OpenSCI: SuperOpTerms, diag_element, biorthogonal_eigen, _all_keys

Random.seed!(1234)

function model(N; amp=0.0)
    L = Lindbladian(N)
    add_hamiltonian!(L, heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, 0.1)
    amp > 0 && add_channel_amplitude_damping!(L, amp)
    return L
end

closest(a, b) = maximum(minimum(abs.(x .- b)) for x in a)   # spectra as sets

@testset "OpenSCI" verbose=true begin

@testset "Lindbladian action" begin
    for N in (2, 3), L in (model(N; amp=0.05), rand(Lindbladian{N}, nH=6, nL=3))
        ρ = rand(PauliSum{N}); O = rand(PauliSum{N})
        @test norm(Matrix(L) * vec(Matrix(ρ)) - vec(Matrix(L * ρ))) < 1e-12
        @test norm(Matrix(L)' * vec(Matrix(O)) - vec(Matrix(L' * O))) < 1e-12
        @test abs(tr(O * (L * ρ)) - tr((L' * O) * ρ)) < 1e-11            # picture invariance
        @test closest(eigvals(Matrix(L)), eigvals(Matrix(L, PauliBasis))) < 1e-11
        t = SuperOpTerms(L)
        for basis in (DyadBasis, PauliBasis)
            keys = _all_keys(basis, N)
            M = Matrix(L, basis)
            @test Matrix(subspace_matrix(L, keys)) ≈ M atol = 1e-12
            @test Matrix(sparse(L, basis)) ≈ M atol = 1e-12
            @test [diag_element(t, k) for k in keys] ≈ diag(M) atol = 1e-12
        end
    end
    # trace preservation: tr(L ρ) = 0
    L = model(3; amp=0.2)
    @test abs(tr(L * rand(PauliSum{3}))) < 1e-12
end

@testset "adjoint terms and left action" begin
    N = 3
    L = model(N; amp=0.05)
    LM = MatrixLindbladian(Matrix(L.H), [Matrix(Lk) for Lk in L.L], L.γ)
    for (Λ, basis) in ((L, DyadBasis), (L, PauliBasis), (LM, DyadBasis))
        t = OpenSCI._terms(Λ)
        keys = OpenSCI._all_keys(Λ, basis)
        M = Matrix(Λ, basis)
        @test Matrix(subspace_matrix(OpenSCI._adjoint(t), keys)) ≈ M' atol = 1e-12
        # (W L)_q over the keys outside a random P, against the dense matrix
        ip = sort(randperm(length(keys))[1:20])
        iq = setdiff(eachindex(keys), ip)
        P = keys[ip]
        W = randn(ComplexF64, 3, length(P))
        out = OpenSCI.external_left_action(t, P, Dict(k => i for (i, k) in enumerate(P)), W)
        WL = W * M[ip, iq]
        @test all(isapprox(get(out, keys[q], zeros(3)), WL[:, a]; atol=1e-12) for (a, q) in enumerate(iq))
        @test all(haskey(out, keys[q]) for (a, q) in enumerate(iq) if norm(WL[:, a]) > 1e-12)
    end
end

@testset "channels and models" begin
    L = Lindbladian(1); add_channel_amplitude_damping!(L, 1.0)
    @test Matrix(L.L[1]) ≈ [0 1; 0 0]                                    # σ⁻ = |0⟩⟨1|
    @test Matrix(heisenberg_1D(3, 1, 1, 1)) ≈ Matrix(heisenberg_1D(3, 1, 1, 1))'
end

@testset "basis conversion" begin
    N = 3
    for O in (rand(PauliSum{N}), Dyad(N, 3, 5), Pauli(N, Y=[1], Z=[3]))
        cP = coefficients(PauliBasis, O); cD = coefficients(DyadBasis, O)
        M = Matrix(O isa Union{Dyad,DyadBasis} ? DyadSum(O) : (O isa PauliSum ? O : PauliSum(O)))
        @test sum(c * Matrix(k) for (k, c) in cP) ≈ M
        @test sum(c * Matrix(k) for (k, c) in cD) ≈ M
    end
end

@testset "biorthogonal eigenvectors" begin
    n, R = 40, 6
    # non-normal, with a 2-fold degenerate eigenvalue straddling the cut
    vals = ComplexF64[0, -0.3, -0.5, -0.5, -0.7 + 0.2im, -0.7 - 0.2im, (-1 .- rand(n - 6) .+ 0.3im .* randn(n - 6))...]
    S = I + 0.4randn(ComplexF64, n, n)
    A = S * Diagonal(vals) / S
    for dense_max in (n, 0)                                              # dense and Krylov paths
        λ, V, W = biorthogonal_eigen(dense_max == 0 ? sparse(A) : A, R; dense_max=dense_max)
        @test norm(W * V - I) < 1e-8
        @test norm(A * V - V * Diagonal(λ)) < 1e-8
        @test norm(W * A - Diagonal(λ) * W) < 1e-8
        @test real(λ) ≈ sort(real(vals), rev=true)[1:R] atol = 1e-8
    end
    # pinv(V) is not a left eigenbasis of a non-normal matrix
    λ, V, W = biorthogonal_eigen(A, R)
    @test norm(pinv(V) * A - Diagonal(λ) * pinv(V)) > 1e-3
end

@testset "selected CI" begin
    N = 3
    L = model(N)
    λex = sort(eigvals(Matrix(L)), by=real, rev=true)
    for basis in (DyadBasis, PauliBasis), selection in (:amplitude, :perturbative)
        res = selected_ci(L, Dyad(N, 0, 0), 2; basis=basis, selection=selection, verbose=false)
        @test res.converged
        @test abs(res.λ[1]) < 1e-10                                     # steady state
        @test closest(res.λ, λex) < 1e-8
        @test norm(res.W * res.V - I) < 1e-10
    end
    # two-sided (left + right) selection and the PT2 criterion
    for basis in (DyadBasis, PauliBasis), (selection, ϵ) in ((:amplitude, 1e-2), (:perturbative, 1e-2), (:pt2, 1e-6))
        res = selected_ci(L, Dyad(N, 0, 0), 2; basis=basis, selection=selection, ϵ_add=ϵ,
                          two_sided=true, verbose=false)
        @test res.converged
        @test closest(res.λ, λex) < 1e-8
        @test norm(res.W * res.V - I) < 1e-10
    end
    # several seed operators, one starting column each
    res = selected_ci(L, [Dyad(N, 0, 0), Dyad(N, 0, 1), Dyad(N, 1, 0)], 2; verbose=false)
    @test closest(res.λ, λex) < 1e-8
    iters = Int[]
    selected_ci(L, Dyad(N, 0, 0), 2; verbose=false, callback=r -> push!(iters, r.iteration))
    @test iters == 1:length(iters)
end

@testset "Löwdin correction" begin
    # regression against the paulis_v2.0 scripts (N=4, dyad basis, converged SCI)
    N = 4
    L = model(N)
    λex = sort(eigvals(Matrix(L)), by=z -> (-real(z), -imag(z)))
    res = selected_ci(L, Dyad(N, 0, 0), 2; verbose=false)
    lw = lowdin_correction(L, res)
    @test abs(res.λ[2] - λex[2]) ≈ 7.107e-5 rtol = 1e-3
    @test abs(lw.λ[2] - λex[2]) ≈ 2.139e-5 rtol = 1e-3
    @test abs(lw.Δ[1]) < 1e-10                                          # steady state untouched
    # complete space ⇒ empty Q ⇒ no correction
    N = 2
    L = model(N)
    full = selected_ci(L, Dyad(N, 0, 0), 4; ϵ_add=0.0, ϵ_clip=0.0, verbose=false)
    @test lowdin_correction(L, full).nQ == 0
    # intruder guard skips tiny denominators
    lw = lowdin_correction(model(4), selected_ci(model(4), Dyad(4, 0, 0), 2; max_iter=1, verbose=false); ϵ_denom=1e-8)
    @test lw.n_skipped[2] > 0 && all(isfinite, lw.λ)
end

@testset "matrix backend" begin
    # same spin model as Pauli strings and as dense matrices
    N = 3
    L = model(N; amp=0.05)
    LM = MatrixLindbladian(Matrix(L.H), [Matrix(Lk) for Lk in L.L], L.γ)
    @test Matrix(LM) ≈ Matrix(L) atol = 1e-12
    ρ = Matrix(rand(PauliSum{N}))
    @test vec(LM * ρ) ≈ Matrix(LM) * vec(ρ)
    rP = selected_ci(L, Dyad(N, 0, 0), 3; verbose=false)      # R = 4 would split a conjugate pair
    rM = selected_ci(LM, (1, 1), 3; verbose=false)
    @test rM.λ ≈ rP.λ atol = 1e-10
    @test length(rM) == length(rP)
    @test lowdin_correction(LM, rM).λ ≈ lowdin_correction(L, rP).λ atol = 1e-10
    ts = 0:0.5:3
    ρ0 = zeros(2^N, 2^N); ρ0[1, 1] = 1
    @test propagate_exact(LM, Matrix(Pauli(N, Z=[2])), ρ0, ts) ≈ propagate_exact(L, Pauli(N, Z=[2]), Dyad(N, 0, 0), ts)

    # dissipative Bose-Hubbard (Westhoff et al.)
    Lb, states = dissipative_bose_hubbard(3, 2; J=1, U=0.7, κ=1.3)
    d = length(states)
    @test d == binomial(3 + 2 - 1, 2)
    @test sum(hopping(states, j, j) for j in 1:3) ≈ 2I                  # particle number
    @test hopping(states, 1, 2)' ≈ hopping(states, 2, 1)
    @test Lb.H ≈ Lb.H'
    ρ = randn(ComplexF64, d, d); ρ = ρ * ρ'; ρ /= tr(ρ)
    @test abs(tr(Lb * ρ)) < 1e-12                                       # trace preserving
    @test vec(Lb * ρ) ≈ Matrix(Lb) * vec(ρ)
    λex = sort(eigvals(Matrix(Lb)), by=z -> (-real(z), -imag(z)))
    full = selected_ci(Lb, [(i, i) for i in 1:d], d^2; ϵ_add=0.0, ϵ_clip=0.0, verbose=false)
    @test closest(full.λ, λex) < 1e-10 && length(full) == d^2
    O = Matrix(hopping(states, 1, 3))
    @test propagate(full, O, ρ, ts) ≈ propagate_exact(Lb, O, ρ, ts) atol = 1e-10

    # momentum (standing-wave) basis: exact many-body rotation, same spectrum
    U = fock_rotation(states, standing_waves(3))
    @test U' * U ≈ I
    Lk, _ = dissipative_bose_hubbard(3, 2; J=1, U=0.7, κ=1.3, basis=:momentum)
    @test closest(eigvals(Matrix(Lk)), λex) < 1e-10
    @test Matrix(Lk.H) ≈ U * Matrix(Lb.H) * U'
    # at U = 0 the Hamiltonian is diagonal in the standing waves
    L0, _ = dissipative_bose_hubbard(4, 2; U=0, basis=:momentum)
    @test norm(L0.H - Diagonal(L0.H)) < 1e-12
end

@testset "dynamics" begin
    N = 3
    L = model(N; amp=0.05)
    ts = 0:0.25:4
    ρ0 = Dyad(N, 0, 0)
    for basis in (DyadBasis, PauliBasis)
        res = selected_ci(L, ρ0, 4^N; basis=basis, ϵ_add=0.0, ϵ_clip=0.0, verbose=false)
        for O in (Pauli(N, Z=[2, 3]), Dyad(N, 1, 1))
            @test propagate(res, O, ρ0, ts) ≈ propagate_exact(L, O, ρ0, ts) atol = 1e-10
        end
    end
    @test propagate_exact(L, Dyad(N, 0, 0), ρ0, ts) ≈ propagate_exact(L, Dyad(N, 0, 0), ρ0, ts; basis=PauliBasis)
    @test real(propagate_exact(L, Pauli(N), ρ0, ts)) ≈ ones(length(ts))  # tr ρ(t) = 1
end

end
