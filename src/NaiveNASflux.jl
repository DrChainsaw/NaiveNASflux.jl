module NaiveNASflux

using Reexport
using NaiveNASlib
using Flux
import Flux.Zygote: hook
using Statistics
using Setfield
using LinearAlgebra
import InteractiveUtils: subtypes
import JuMP: @variable, @constraint, @expression, SOS1

export convinputvertex, denseinputvertex, rnninputvertex, fluxvertex, concat

export ActivationContribution, NeuronValueEvery, LazyMutable

export layer, layertype

export KernelSizeAligned, mutate_weights

include("types.jl")
include("util.jl")
include("constraints.jl")
include("select.jl")
include("mutable.jl")
include("vertex.jl")
include("pruning.jl")
include("weightinit.jl")

# Stuff to integrate with Flux and Zygote
include("functor.jl")
include("zygote.jl")

# Reexporting before include("functor.jl") causes a warning about duplicate name (flatten) in NaiveNASlib and Flux when subtypes are called
# https://discourse.julialang.org/t/avoid-error-message-for-function-name-conflict/37176/10
@reexport using NaiveNASlib
@reexport using Flux

end # module
