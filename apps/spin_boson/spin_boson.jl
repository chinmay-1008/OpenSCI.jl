# Spin–boson toy problems.
#
#   julia --project=apps apps/spin_boson/spin_boson.jl
#
# run():  a driven, damped qubit. ⟨σᶻ⟩(t) and purity from the exact propagator (the old
#         version Euler-stepped ρ), then the steady state from selected CI.
# run2(): spectrum of the spin–boson Hamiltonian with the boson mode encoded in N qubits
#         (binary encoding, PauliOperators' `boson_to_paulis`), and ⟨σᶻ⟩ of its eigenstates.
#         https://www.uni-ulm.de/fileadmin/website_uni_ulm/nawi.inst.260/%C3%9Cbungen/6Spin_Boson_Modell.pdf
using OpenSCI, PauliOperators, LinearAlgebra, Printf, Plots

function run(; ω0=1.0, γ=0.1, T=20.0)
    ρ0 = Dyad(1, 0, 0)
    @printf(" Initial density: %s\n", ρ0)

    H = ω0 * PauliSum(Pauli("X"))
    σ⁻ = 0.5 * (Pauli("X") - 1im * Pauli("Y"))      # |1⟩⟨0|
    σᶻ = PauliSum(Pauli("Z"))
    L = Lindbladian(H, [σ⁻, σᶻ], [γ, γ])
    display(L)

    ts = range(0, T, length=2001)
    X, Y, Z = (real(propagate_exact(L, Pauli(s), ρ0, ts)) for s in ("X", "Y", "Z"))
    purity = @. (1 + X^2 + Y^2 + Z^2) / 2                # tr ρ² for one qubit

    p = plot(ts, Z, label="⟨σᶻ⟩", xlabel="time", lw=2, size=(640, 380), framestyle=:box)
    plot!(p, ts, purity, label="tr ρ²", lw=2)
    out = joinpath(@__DIR__, "spin_dynamics.pdf")
    savefig(p, out)
    println(" wrote ", out)

    println("\n Steady state via diagonalization (selected CI, R = 1):")
    res = selected_ci(L, ρ0, 1; ϵ_add=0.0, ϵ_clip=0.0, verbose=false)
    display(res)
    zss = real(propagate(res, Pauli("Z"), ρ0, [1e6])[1])
    @printf(" ⟨σᶻ⟩_ss = % .8f   (dynamics at t = %.0f: % .8f)\n", zss, T, Z[end])
    return res
end

# `p'` on a PauliSum is a lazy Adjoint, which ⊗ does not take; Pauli strings are Hermitian,
# so the adjoint just conjugates the coefficients.
dag(p::PauliSum{N,T}) where {N,T} = PauliSum{N,T}(k => conj(c) for (k, c) in p)

function build_full_hamiltonian(ω, λ, g, N)
    b⁺ = boson_to_paulis(N)                              # raising operator, N qubits
    b = dag(b⁺)
    Hs = ω / 2 * PauliSum(Pauli(1, Z=[1]))
    Hb = ω * b⁺ * b
    H = Hs ⊕ Hb

    σᶻ = PauliSum(Pauli("Z"))
    σ⁺ = 0.5 * (Pauli("X") + 1im * Pauli("Y"))
    σ⁻ = dag(σ⁺)
    Hc = g * (σ⁺ ⊗ b + σ⁻ ⊗ b⁺)
    Hc += λ * σᶻ ⊗ (b⁺ + b)
    coeff_clip!(Hc, 1e-12)
    return H + Hc
end

function run2(; ω0=1.0, N=4, g=0.1, λ=0.1, nstates=5)
    H = build_full_hamiltonian(ω0, λ, g, N)
    e, v = eigen(Hermitian(Matrix(H)))
    println("\n Spin–boson eigenvalues (boson in $N qubits, $(2^N) Fock states)")
    for i in 1:min(length(e), nstates)
        @printf(" %3i %12.8f\n", i, e[i])
    end
    Z = Matrix(Pauli("Z"))
    for i in 1:min(length(e), nstates)
        ψ = reshape(v[:, i], 2, 2^N)                     # qubit 1 (spin) is the fastest index
        ρs = ψ * ψ'                                      # reduced spin density matrix
        @printf(" state %d: tr(σᶻ ρ_spin) = % .8f\n", i, real(tr(ρs * Z)))
    end
    return e, v
end

run()
run2()
