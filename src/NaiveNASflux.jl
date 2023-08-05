module NaiveNASflux

using Reexport
@reexport using NaiveNASlib
using NaiveNASlib.Extend, NaiveNASlib.Advanced
import Flux
using Flux: Dense, Conv, ConvTranspose, CrossCor, LayerNorm, BatchNorm, InstanceNorm, GroupNorm, 
            MaxPool, MeanPool, Dropout, AlphaDropout, GlobalMaxPool, GlobalMeanPool, cpu
import Optimisers
import Functors
using Functors: @functor
using Statistics
using Setfield: @set, setproperties
using LinearAlgebra
import JuMP
import JuMP: @variable, @constraint, @expression, SOS1, MOI
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


using PrecompileTools

@setup_workload begin
    iv1 = denseinputvertex("iv1", 1)
    v1 = fluxvertex("v1", Dense(nout(iv1) => 1), iv1)
    v2 = concat("v2", v1, v1; layerfun=ActivationContribution)
    v3 = concat("v3", v2,v1,iv1)
    v4 = "v4" >> v3 + v3
    v5 = "v5" >> v4 + v4 + v4
    v6 = fluxvertex("v6", Dense(nout(v5) => 1), v5; layerfun = ActivationContribution ∘ LazyMutable)

    g1 = CompGraph(iv1, v6)
    x1 = ones(Float32, 1, 1)

    @compile_workload begin
        iv1 = denseinputvertex("iv1", 1)
        fluxvertex("v1", Dense(nout(iv1) => 1), iv1)

        g1(x1)
        Flux.@code_adjoint g1(x1)
        #Optimisers.setup(Optimisers.Descent(0.1f0), g1)
        #Flux.gradient((g,x) -> sum(g(x)), g1, x1)
    end
end

end # module
