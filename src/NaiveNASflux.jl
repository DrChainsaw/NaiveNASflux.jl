module NaiveNASflux

using LinearAlgebra: length
using Base: IteratorEltype, @var
using Reexport
using NaiveNASlib
import NaiveNASlib: DecoratingJuMPΔSizeStrategy
using Flux
import Flux.Zygote: hook
using Statistics
using Setfield
using LinearAlgebra
import InteractiveUtils: subtypes
import JuMP: @variable, @constraint

export FluxLayer, FluxParLayer, FluxDense, FluxRecurrent, FluxRnn, FluxGru, FluxLstm, FluxConvolutional, FluxConv, FluxConvTranspose, FluxDepthwiseConv, FluxCrossCor, FluxTransparentLayer, FluxParInvLayer, FluxDiagonal, FluxLayerNorm, FluxParNorm, FluxBatchNorm, FluxInstanceNorm, FluxGroupNorm, FluxNoParLayer

export mutable, concat, AbstractMutableComp, MutableLayer, LazyMutable, NoParams

export named, validated, logged

export ActivationContribution, neuron_value, neuronvaluetaylor, Ewma, NeuronValueEvery

export indim, outdim, actdim, layer, layertype

export idmapping, idmapping_nowarn

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
