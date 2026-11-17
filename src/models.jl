"""
    heisenberg_1D(N, Jx, Jy, Jz; periodic=true)

`H = -2 Σᵢ (Jx XᵢXᵢ₊₁ + Jy YᵢYᵢ₊₁ + Jz ZᵢZᵢ₊₁)` on a ring (or open chain).
"""
function heisenberg_1D(N, Jx, Jy, Jz; periodic=true)
    H = PauliSum(N)
    for i in 1:(periodic ? N : N - 1)
        j = i % N + 1
        for (J, P) in ((Jx, Pauli(N, X=[i, j])), (Jy, Pauli(N, Y=[i, j])), (Jz, Pauli(N, Z=[i, j])))
            k = PauliBasis(P)
            H[k] = get(H, k, 0.0im) - 2J * coeff(P)
        end
    end
    return H
end
