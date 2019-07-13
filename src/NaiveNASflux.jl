module NaiveNASflux

using Reexport
@reexport using NaiveNASlib
@reexport using Flux
import Flux.Tracker: hook
using Statistics
using Setfield
using LinearAlgebra

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

end # module
