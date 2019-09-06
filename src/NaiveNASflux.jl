module NaiveNASflux

using Reexport
@reexport using NaiveNASlib
@reexport using Flux
import Flux.Tracker: hook
using Statistics
using Setfield
using LinearAlgebra
import InteractiveUtils: subtypes

export FluxLayer, FluxParLayer, FluxDense, FluxRecurrent, FluxRnn, FluxGru, FluxLstm, FluxConvolutional, FluxConv, FluxConvTranspose, FluxDepthwiseConv, FluxTransparentLayer, FluxParInvLayer, FluxDiagonal, FluxLayerNorm, FluxParNorm, FluxBatchNorm, FluxInstanceNorm, FluxGroupNorm, FluxNoParLayer

export mutable, concat, AbstractMutableComp, MutableLayer, LazyMutable, NoParams

export named, validated, logged

export ActivationContribution, neuron_value

export nin, nout, indim, outdim, actdim, layer, layertype

export idmapping


include("types.jl")
include("util.jl")
include("mutable.jl")
include("vertex.jl")
include("pruning.jl")
include("weightinit.jl")


# Modified version of Flux.treelike for mutable structs which are mutable mainly because they are intended to be wrapped in MutationVertices which are not easy to create in the manner which Flux.mapchildren is designed.
function treelike_mutable(m::Module, T, fs = treelike_fields(T))
  @eval m begin
    Flux.children(x::$T) = ($([:(x.$f) for f in fs]...),)
    function Flux.mapchildren(f, x::$T)
        $([:(x.$fn = f(x.$fn)) for fn in fs]...)
        return x
    end
  end
end

treelike_fields(T) = fieldnames(T)

macro treelike_mutable(T, fs = nothing)
  fs == nothing || Meta.isexpr(fs, :tuple) || error("@treelike_mutable T (a, b)")
  fs = fs == nothing ? [] : [:($(map(QuoteNode, fs.args)...),)]
  :(treelike_mutable(@__MODULE__, $(esc(T)), $(fs...)))
end

for subtype in subtypes(AbstractMutableComp)
  @treelike_mutable subtype
end

end # module
