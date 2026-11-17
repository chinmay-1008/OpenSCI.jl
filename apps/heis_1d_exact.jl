# Reference: the 5 eigenvalues with largest real part of the full Lindbladian, by Arnoldi on
# the sparse 4^N × 4^N superoperator (replaces the old Arpack + LinearMap version).
#
#   julia --project=apps apps/heis_1d_exact.jl [N=6]
using OpenSCI, PauliOperators, SparseArrays, KrylovKit, Printf

function run(; N=6, nev=5)
    L = Lindbladian(N)
    add_hamiltonian!(L, heisenberg_1D(N, 1.1, 1.2, 1.3))
    add_channel_dephasing!(L, 0.1)
    add_channel_depolarizing!(L, 0.1)
    add_channel_amplitude_damping!(L, 0.1)
    display(L)

    @time A = sparse(L)
    @printf(" sparse L: %d × %d, %d nonzeros\n", size(A)..., nnz(A))
    @time vals, vecs, info = eigsolve(A, nev, :LR; tol=1e-9, krylovdim=120, maxiter=300)
    @printf(" converged %d of %d\n", info.converged, nev)
    for (i, λ) in enumerate(vals[1:nev])
        @printf(" %3i %14.10f %14.10fi\n", i, real(λ), imag(λ))
    end
    return L, vals, vecs
end

run(N=parse(Int, get(ARGS, 1, "6")))
