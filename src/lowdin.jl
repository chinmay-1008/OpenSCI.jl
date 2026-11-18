"""
    lowdin_correction(L, res::SCIResult; ϵ_denom=0.0) -> NamedTuple

Second-order Löwdin (Brillouin–Wigner) correction for every eigenvalue of an SCI result,
using a diagonal `L_QQ` over the external space `Q` (keys reached by `L` from `P`):

    λᵢ⁽²⁾ = λᵢ + Σ_q (wᵢ L_Pq)(L_qP vᵢ) / (λᵢ − L_qq)

with `wᵢ` the biorthogonal left eigenvectors from the SCI result (`W*V = I`). This is the
same number as `wᵢ L_eff vᵢ` with `L_eff = L_PP + L_PQ (λᵢ − D_QQ)⁻¹ L_QP`.

Near-degenerate denominators (intruder states) make the series blow up. Terms with
`|λᵢ − L_qq| < ϵ_denom` are skipped and counted; the default `0` keeps everything.

Returns `(λ, Δ, nQ, min_denominator, n_skipped)` with per-state vectors.
"""
function lowdin_correction(Λ::AbstractLindbladian, res::SCIResult; ϵ_denom=0.0)
    t = _terms(Λ)
    R = nstates(res)
    ext = external_action(t, res.basis, res.pos, res.V)    # q => (L_qP V)
    Δ = zeros(ComplexF64, R)
    mind = fill(Inf, R)
    nskip = zeros(Int, R)
    wLq = zeros(ComplexF64, R)
    for (q, LqV) in ext
        # one pass over L(q): diagonal L_qq and the column L_Pq contracted with W
        Lqq = 0.0im
        fill!(wLq, 0)
        foreach_image(t, q) do k, c
            if k == q
                Lqq += c
            else
                i = get(res.pos, k, 0)
                i > 0 && (wLq .+= c .* @view res.W[:, i])
            end
        end
        for i in 1:R
            d = res.λ[i] - Lqq
            mind[i] = min(mind[i], abs(d))
            if abs(d) < ϵ_denom
                nskip[i] += 1
                continue
            end
            Δ[i] += wLq[i] * LqV[i] / d
        end
    end
    return (λ=res.λ .+ Δ, Δ=Δ, nQ=length(ext), min_denominator=mind, n_skipped=nskip)
end
