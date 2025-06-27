module OpenSCI

using PauliOperators
using LinearAlgebra
using Printf
using LinearMaps

abstract type SuperOperator{N} end

include("type_SparseDyadVectors.jl")
include("type_Lindbladian.jl")
include("type_WeightDissipator.jl")
include("type_SubspaceDissipator.jl")
include("methods.jl")
include("sci.jl")
include("Hamiltonians.jl")
include("function_dot.jl")
include("index.jl")

export Lindbladian
export SubspaceDissipator 
export SparseDyadVectors
export todense
export selected_ci
export multiply
export build_subspace_L
export add_hamiltonian! 
export add_channel_amplitude_damping! 
export add_channel_dephasing! 
export add_channel_depolarizing! 
export PauliMatrix
export index
export evolve
export build_subspace_L_generalized

end
