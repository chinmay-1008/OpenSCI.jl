using OpenSCI
using PauliOperators
using LinearAlgebra
# using DifferentialEquations
using Plots, Measures
using Printf
using OrderedCollections
using StatProfilerHTML
# using BenchmarkTools
# using Arpack
gr(display_type=:inline)

function run()
    N = 2
    dim = 2^N


    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)  
    add_channel_depolarizing!(L, .1)
    # println("Lindbladian: ")
    # display(L)

    Lmat = Matrix(L)
    dim_L = size(Lmat, 1)
    # println("Diagonalization started")

    F = eigen(Lmat)
    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states_F = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]


    @printf("\n Eigenvalues of L in dyad basis:\n")
    # for i in (length(F.values) - 2):length(F.values)
    for i in 1:length(F.values)

        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states_F[i])))
    end

    U = OpenSCI.Dyad2Pauli_rotation(N)
    Lpauli = U * Lmat * U'

    P = eigen(Lpauli)
    # Sort Eigenvalues by real part
    perm = sortperm(P.values, by=real)
    P.values .= P.values[perm]
    P.vectors .= P.vectors[:, perm]
    states_P = [reshape(P.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(P.values)]

    @printf("\n Eigenvalues of L in pauli basis:\n")
    # for i in (length(P.values) - 2):length(P.values)
    for i in 1:length(P.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(P.values[i]), imag(P.values[i]), real(tr(states_P[i])))
    end

    return


    for j in (length(F.values) - 1):(length(F.values)-1)
        v = F.vectors[:, j]

        # --- dyad coefficients
        coeff_dyad = abs.(v)
        coeff_dyad_sorted = sort(coeff_dyad; rev=true)

        # --- pauli coefficients
        coeff_pauli = abs.(U' * v)   # since U is unitary if normalized
        coeff_pauli_sorted = sort(coeff_pauli; rev=true)

        plt = plot(coeff_dyad_sorted,
                   label="Dyad basis coefficients",
                   xlabel="Coefficient index (sorted)",
                   ylabel="|c|",
                   title="N = $N, Eigenvector for eigenvalue = $(round(F.values[j], digits=4))",
                   dpi = 300, lw=2)

        plot!(plt, coeff_pauli_sorted,
              label="Pauli basis coefficients",lw=2 )

    end

    savefig("test/eigenvectors.png")
end

function pauli_basis()
    N = 4

    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)  
    add_channel_depolarizing!(L, .1)

    Lmat = Matrix(L)

    Lpauli = zeros(ComplexF64, (4^N, 4^N))

    for z in 0:2^N-1
        for x in 0:2^N-1
            p = PauliBasis(Pauli(z, x, N))
            σ = L*p

            for (qi, ci) in σ
                Lpauli[index(qi), index(p)] = ci
            end
        end
    end

    F = eigen(Lmat)
    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states_F = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]


    @printf("\n Eigenvalues of L in dyad basis:\n")
    # for i in (length(F.values) - 2):length(F.values)
    for i in 1:length(F.values)

        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states_F[i])))
    end

    P = eigen(Lpauli)
    # Sort Eigenvalues by real part
    perm = sortperm(P.values, by=real)
    P.values .= P.values[perm]
    P.vectors .= P.vectors[:, perm]

    states_P = [reshape(P.vectors[:,i], 2^N, 2^N) for i in 1:length(P.values)]

    # states_P = []

    # for i in 1:length(P.values)
    #     rho = zeros(ComplexF64, 2^N, 2^N)
    #     vec_state = P.vectors[:, i]
    #     for z in 0:2^N-1
    #         for x in 0:2^N-1
    #             p = Pauli(z, x, N) * (1/2^N) # normalized Pauli
    #             rho += vec_state[index(p)] * Matrix(p)
    #         end
    #     end
    #     push!(states_P, rho)
    # end
    # display(states_P[end-3])

    @printf("\n Eigenvalues of L in pauli basis:\n")
    # for i in (length(P.values) - 2):length(P.values)
    for i in 1:length(P.values)

        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(P.values[i]), imag(P.values[i]), real(tr(states_P[i])))
    end

    # return
    
    # for j in (length(F.values)):(length(F.values))
    j = length(F.values) - 2

    v = F.vectors[:, j]

    u = P.vectors[:, j]
    # --- dyad coefficients
    coeff_dyad = abs.(v)
    coeff_dyad_sorted = sort(coeff_dyad; rev=true)

    # --- pauli coefficients
    coeff_pauli = abs.(u)   
    coeff_pauli_sorted = sort(coeff_pauli; rev=true)

    idx = length(F.values) - j
    plt = plot(coeff_dyad_sorted,
                label="Dyad basis coefficients",
                xlabel="Coefficient index (sorted)",
                ylabel="|c|",
                title="N = $N, eigenvalue(idx_$idx) = $(round(F.values[j], digits=4))",
                dpi = 300, lw=2)

    plot!(plt, coeff_pauli_sorted,
            label="Pauli basis coefficients",lw=2 )

    # end

    savefig("test/eigenvectors_fast.png")
    return
end

function run_sci_pauli()

    N = 5

    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)  
    add_channel_depolarizing!(L, .1)

    Lmat = Matrix(L)

    Lpauli = zeros(ComplexF64, (4^N, 4^N))

    for z in 0:2^N-1
        for x in 0:2^N-1
            p = PauliBasis(Pauli(z, x, N))
            σ = L*p

            for (qi, ci) in σ
                Lpauli[index(qi), index(p)] = ci
            end
        end
    end

    F = eigen(Lpauli)
    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states_F = [reshape(F.vectors[:,i], 2^N, 2^N) for i in 1:length(F.values)]
    display(states_F[end])

    @printf("\n Eigenvalues of L in Pauli basis:\n")
    for i in (length(F.values) - 2):length(F.values)
    # for i in 1:length(F.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states_F[i])))
    end

    # state = PauliSum(rand(Pauli{N}))
    state = Pauli(0, 0, N) + Pauli(0, 1, N) + Pauli(1, 0, N)

    nkeep = 2
    v0 = SparsePauliVectors(state, R = nkeep)

    p_pauli, eig_sci = selected_ci(L, v0, ϵdiscard=1e-3, max_iter_outer = 10)
    println("Eigenvalues after SCI using PAULI basis")

    display(eig_sci)
end

function run_sci_dyad()

    N = 5
    dim = 2^N


    # Initializing the Lindbladian
    L = Lindbladian(N)
    add_hamiltonian!(L, OpenSCI.heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)  
    add_channel_depolarizing!(L, .1)
    # println("Lindbladian: ")
    # display(L)

    Lmat = Matrix(L)
    dim_L = size(Lmat, 1)
    # println("Diagonalization started")

    F = eigen(Lmat)
    # Sort Eigenvalues by real part
    perm = sortperm(F.values, by=real)
    F.values .= F.values[perm]
    F.vectors .= F.vectors[:, perm]
    states_F = [reshape(F.vectors[:,i], 2^N, 2^N)/sqrt(2^N) for i in 1:length(F.values)]

    @printf("\n Eigenvalues of L in Pauli basis:\n")
    for i in (length(F.values) - 2):length(F.values)
    # for i in 1:length(F.values)
        @printf(" %4i %12.8f %12.8fi Tr = %12.8f\n", i, real(F.values[i]), imag(F.values[i]), real(tr(states_F[i])))
    end


    state = Dyad(N, 0, 0) + Dyad(N, 0, 1) + Dyad(N, 1, 0)
    nkeep = 2
    v0 = SparseDyadVectors(state, R = nkeep)

    p_dyad, eig_sci = selected_ci(L, v0, ϵdiscard=1e-3, max_iter_outer = 10)
    println("Eigenvalues after SCI using DYAD basis")
    display(eig_sci)
end

# run_sci_pauli()
run_sci_dyad()
# pauli_basis()