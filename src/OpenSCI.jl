"""
    OpenSCI

Selected-CI eigensolver for Lindbladian superoperators, with a second-order Löwdin
correction for the truncated space and eigen-expansion dynamics.

The same code runs in either operator basis, chosen by the key type:
`DyadBasis{N}` (|k⟩⟨b|) or `PauliBasis{N}` (Pauli strings).
"""
module OpenSCI

using PauliOperators
using LinearAlgebra
using SparseArrays
using Printf
import KrylovKit

include("lindbladian.jl")
include("models.jl")
include("basis.jl")
include("eigen.jl")
include("sci.jl")
include("lowdin.jl")
include("dynamics.jl")

export Lindbladian, add_hamiltonian!
export add_channel_dephasing!, add_channel_depolarizing!, add_channel_amplitude_damping!
export heisenberg_1D
export SuperOpTerms, subspace_matrix, coefficients
export SCIResult, selected_ci
export lowdin_correction
export propagate, propagate_exact

end
