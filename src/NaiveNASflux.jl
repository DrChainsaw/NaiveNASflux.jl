module NaiveNASflux

using Reexport
@reexport using NaiveNASlib
using NaiveNASlib.Extend, NaiveNASlib.Advanced
import Flux
using Flux: Dense, Conv, ConvTranspose, CrossCor, LayerNorm, BatchNorm, InstanceNorm, GroupNorm, 
            MaxPool, MeanPool, Dropout, AlphaDropout, GlobalMaxPool, GlobalMeanPool, cpu
import Functors
using Functors: @functor
using Statistics
using Setfield: @set, setproperties
using LinearAlgebra
import JuMP: @variable, @constraint, @expression, SOS1
import ChainRulesCore
import ChainRulesCore: rrule_via_ad, RuleConfig, HasReverseMode, Tangent, NoTangent

export denseinputvertex, rnninputvertex, fluxvertex, concat

export convinputvertex, conv1dinputvertex, conv2dinputvertex, conv3dinputvertex

export ActivationContribution, LazyMutable

export layer

export KernelSizeAligned

export NeuronUtilityEvery

include("types.jl")
include("util.jl")
include("constraints.jl")
include("select.jl")
include("mutable.jl")
include("vertex.jl")
include("neuronutility.jl")

# Stuff to integrate with Zygote
include("chainrules.jl")

end # module
