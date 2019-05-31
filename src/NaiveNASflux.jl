module NaiveNASflux

using NaiveNASlib
using Flux

export MutableLayer

export mutable

export nin, nout, indim, outdim

# Types of layers parameterized with weights and biases and with similar handling
# w.r.t what weights and biases means in terms of number of inputs and number of outputs
ParDense = Dense
ParConv = Union{Conv, ConvTranspose, DepthwiseConv} # CrossCor missing, but I can't seem to import it...
ParRnn = Flux.Recur
ParLayer = Union{ParDense, ParRnn, ParConv}

# Invariant layers with parameters, i.e nin == nout always and parameter selection must
# be performed
ParInvLayer = Union{Flux.Diagonal, LayerNorm, BatchNorm, InstanceNorm, GroupNorm}

# Transparent layers, i.e nin == nout always and there are not parameters
TransparentLayer = Union{MaxPool, MeanPool, Dropout, AlphaDropout}

# Compositions? Might not have any common methods...
# MaxOut, Chain

include("util.jl")
include("mutable.jl")
include("vertex.jl")


end # module
