module NaiveNASflux

using Reexport
@reexport using NaiveNASlib
using NaiveNASlib.Extend, NaiveNASlib.Advanced
import Flux
using Flux: Dense, Conv, ConvTranspose, DepthwiseConv, CrossCor, LayerNorm, BatchNorm, InstanceNorm, GroupNorm, 
            MaxPool, MeanPool, Dropout, AlphaDropout, GlobalMaxPool, GlobalMeanPool, cpu
import Functors
using Functors: @functor
using Statistics
using Setfield: @set, setproperties
using LinearAlgebra
import JuMP: @variable, @constraint, @expression, SOS1

export denseinputvertex, rnninputvertex, fluxvertex, concat

export convinputvertex, conv1dinputvertex, conv2dinputvertex, conv3dinputvertex

export ActivationContribution, LazyMutable

export layer

export KernelSizeAligned

export neuron_value, NeuronValueEvery

include("types.jl")
include("util.jl")
include("constraints.jl")
include("select.jl")
include("mutable.jl")
include("vertex.jl")
include("neuronvalue.jl")
include("weightinit.jl")

# Stuff to integrate with Zygote
include("zygote.jl")

end # module
