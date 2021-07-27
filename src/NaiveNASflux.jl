module NaiveNASflux

using Reexport
@reexport using NaiveNASlib
import Flux
using Flux: Dense, Conv, ConvTranspose, DepthwiseConv, CrossCor, LayerNorm, BatchNorm, InstanceNorm, GroupNorm, 
            MaxPool, MeanPool, Dropout, AlphaDropout, GlobalMaxPool, GlobalMeanPool, cpu
using Functors: @functor
using Statistics
using Setfield: @set, setproperties
using LinearAlgebra
import InteractiveUtils: subtypes
import JuMP: @variable, @constraint, @expression, SOS1

export denseinputvertex, rnninputvertex, fluxvertex, concat

export convinputvertex, conv1dinputvertex, conv2dinputvertex, conv3dinputvertex

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

end # module
