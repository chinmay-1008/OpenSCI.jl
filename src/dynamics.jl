"""
    propagate(res::SCIResult, O, ρ0, times) -> Vector{ComplexF64}

`tr(O ρ(t))` from the SCI eigen-expansion

    ρ(t) ≈ Σᵢ e^{λᵢ t} vᵢ (wᵢ · ρ0)

`O` and `ρ0` are PauliOperators objects (e.g. `Pauli(N, Z=[1])`, `Dyad(N,0,0)`), or plain
matrices for a [`MatrixLindbladian`](@ref); the part of `ρ0` outside the P space is dropped. Each mode's overlap with `O` is computed once, so
long time grids are cheap.
"""
function propagate(res::SCIResult{K}, O, ρ0, times) where K
    r0 = coefficients(_basis_type(K), ρ0)
    c = res.W * [get(r0, k, 0.0im) for k in res.basis]                 # wᵢ · ρ0
    o = transpose(res.V) * [_trace_with(O, k) for k in res.basis]       # tr(O vᵢ)
    return [sum(@. exp(res.λ * t) * o * c) for t in times]
end

"""
    propagate_exact(L, O, ρ0, times; basis=DyadBasis) -> Vector{ComplexF64}

Reference `tr(O e^{Lt} ρ0)` from dense diagonalization of `L` (`4^N × 4^N`, so small N only).
"""
function propagate_exact(Λ::AbstractLindbladian, O, ρ0, times; basis::Type=DyadBasis)
    keys = _all_keys(Λ, basis)
    F = eigen(Matrix(Λ, basis))
    r0 = coefficients(_basis_type(eltype(keys)), ρ0)
    c = F.vectors \ [get(r0, k, 0.0im) for k in keys]
    o = transpose(F.vectors) * [_trace_with(O, k) for k in keys]
    return [sum(@. exp(F.values * t) * o * c) for t in times]
end
