
abstract type FluxLayer end

layertype(l) = NaiveNASlib.shapetrait(l)
layer(l) = l

# Types for layers with parameters (e.g. weights and biases) and with similar handling
# w.r.t what shape of parameters means in terms of number of inputs and number of outputs
abstract type FluxParLayer <: FluxLayer end

abstract type Flux2D <: FluxParLayer end
struct GenericFlux2D <: Flux2D end
struct FluxDense <: Flux2D end
NaiveNASlib.shapetrait(::Dense) = FluxDense()

# Might be a Flux2D, but not exactly the same. Also want to leave the door open to when/if they accept 3D input too
# Type hierarchies are hard :/
abstract type FluxRecurrent <:FluxParLayer end
struct GenericFluxRecurrent <: FluxRecurrent end
struct FluxRnn <: FluxRecurrent end
struct FluxLstm <: FluxRecurrent end
struct FluxGru <: FluxRecurrent end
NaiveNASlib.shapetrait(::Flux.Recur{<:Flux.RNNCell}) = FluxRnn()
NaiveNASlib.shapetrait(::Flux.Recur{<:Flux.LSTMCell}) = FluxLstm()
NaiveNASlib.shapetrait(::Flux.Recur{<:Flux.GRUCell}) = FluxGru()

abstract type FluxConvolutional{N} <: FluxParLayer end
struct GenericFluxConvolutional{N} <: FluxConvolutional{N} end
struct FluxConv{N} <: FluxConvolutional{N} end
struct FluxConvTranspose{N}  <: FluxConvolutional{N} end
struct FluxDepthwiseConv{N} <: FluxConvolutional{N}  end
struct FluxCrossCor{N} <: FluxConvolutional{N}  end
NaiveNASlib.shapetrait(::Conv{N}) where N = FluxConv{N}()
NaiveNASlib.shapetrait(::ConvTranspose{N}) where N = FluxConvTranspose{N}()
NaiveNASlib.shapetrait(::DepthwiseConv{N}) where N = FluxDepthwiseConv{N}()
NaiveNASlib.shapetrait(::CrossCor{N}) where N = FluxCrossCor{N}()


abstract type FluxTransparentLayer <: FluxLayer end
# Invariant layers with parameters, i.e nin == nout always and parameter selection must
# be performed
abstract type FluxParInvLayer <: FluxTransparentLayer end
struct FluxDiagonal <: FluxParInvLayer end
struct FluxLayerNorm <: FluxParInvLayer end
abstract type FluxParNorm <: FluxParInvLayer end
struct FluxBatchNorm <: FluxParNorm end
struct FluxInstanceNorm <: FluxParNorm end
struct FluxGroupNorm <: FluxParNorm end

NaiveNASlib.shapetrait(::Flux.Diagonal) = FluxDiagonal()
NaiveNASlib.shapetrait(::LayerNorm) = FluxLayerNorm()
NaiveNASlib.shapetrait(::BatchNorm) = FluxBatchNorm()
NaiveNASlib.shapetrait(::InstanceNorm) = FluxInstanceNorm()
NaiveNASlib.shapetrait(::GroupNorm) = FluxGroupNorm()

# Transparent layers, i.e nin == nout always and there are no parameters
abstract type FluxNoParLayer <: FluxTransparentLayer end
struct FluxPoolLayer <: FluxNoParLayer end
struct FluxDropOut <: FluxNoParLayer end

layertype(l::MaxPool) = FluxPoolLayer()
layertype(l::MeanPool) = FluxPoolLayer()
layertype(l::Dropout) = FluxDropOut()
layertype(l::AlphaDropout) = FluxDropOut()
layertype(l::GlobalMaxPool) = FluxPoolLayer()
layertype(l::GlobalMeanPool) = FluxPoolLayer()

# Compositions? Might not have any common methods...
# MaxOut, Chain?
