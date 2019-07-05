module NaiveNASflux

using NaiveNASlib
using Flux
using Setfield

export layertype

export FluxLayer, FluxParLayer, FluxDense, FluxRecurrent, FluxRnn, FluxGru, FluxLstm, FluxConvolutional, FluxConv, FluxConvTranspose, FluxDepthwiseConv, FluxTransparentLayer, FluxParInvLayer, FluxDiagonal, FluxLayerNorm, FluxParNorm, FluxBatchNorm, FluxInstanceNorm, FluxGroupNorm, FluxNoParLayer

export mutable, concat, MutableLayer, LazyMutable, NoParams

export nin, nout, indim, outdim, actdim, layer

include("types.jl")
include("util.jl")
include("mutable.jl")
include("vertex.jl")

end # module
