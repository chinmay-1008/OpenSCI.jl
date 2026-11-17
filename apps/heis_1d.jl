# Selected CI for the two slowest modes of a noisy Heisenberg ring (N = 8, dim L = 4^8).
#
#   julia --project=apps apps/heis_1d.jl [N=8] [dyad|pauli] [max_iter=4]
#
# max_iter = 4 is the old app's setting; it stops before the space converges.
using OpenSCI, PauliOperators, Printf

function run(; N=8, basis=DyadBasis, max_iter=4)
    L = Lindbladian(N)
    add_hamiltonian!(L, heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, 0.1)
    add_channel_amplitude_damping!(L, 0.1)     # σ⁻ = (X+iY)/2: γ is the physical rate
    display(L)

    # seed with |0⟩⟨0|, |0⟩⟨1|, |1⟩⟨0| as separate starting vectors (the old `eye!` seeding)
    seeds = [Dyad(N, 0, 0), Dyad(N, 0, 1), Dyad(N, 1, 0)]
    @time res = selected_ci(L, seeds, 2; basis=basis, ϵ_clip=1e-5, ϵ_add=sqrt(1e-5), max_iter=max_iter)
    display(res)
    return res
end

N = parse(Int, get(ARGS, 1, "8"))
basis = lowercase(get(ARGS, 2, "dyad")) == "pauli" ? PauliBasis : DyadBasis
run(N=N, basis=basis, max_iter=parse(Int, get(ARGS, 3, "4")))
