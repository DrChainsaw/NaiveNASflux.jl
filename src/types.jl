
abstract type FluxLayer end

layertype(l::T) where T = T

# Types for layers with parameters (e.g. weights and biases) and with similar handling
# w.r.t what shape of parameters means in terms of number of inputs and number of outputs
abstract type FluxParLayer <: FluxLayer end
struct FluxDense <: FluxParLayer end
layertype(l::Dense) = FluxDense()

abstract type FluxRecurrent <:FluxParLayer end
struct FluxRnn <: FluxRecurrent end
struct FluxLstm <: FluxRecurrent end
struct FluxGru <: FluxRecurrent end
layertype(l::Flux.Recur{<:Flux.RNNCell}) = FluxRnn()
layertype(l::Flux.Recur{<:Flux.LSTMCell}) = FluxLstm()
layertype(l::Flux.Recur{<:Flux.GRUCell}) = FluxGru()

abstract type FluxConvolutional{N} <: FluxParLayer end
struct FluxConv{N} <: FluxConvolutional{N} end
struct FluxConvTranspose{N}  <: FluxConvolutional{N} end
struct FluxDepthwiseConv{N} <: FluxConvolutional{N}  end
struct FluxCrossCor{N} <: FluxConvolutional{N}  end
layertype(l::Conv{N}) where N = FluxConv{N}()
layertype(l::ConvTranspose{N}) where N = FluxConvTranspose{N}()
layertype(l::DepthwiseConv{N}) where N = FluxDepthwiseConv{N}()
layertype(l::CrossCor{N}) where N = FluxCrossCor{N}()


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

layertype(l::Flux.Diagonal) = FluxDiagonal()
layertype(l::LayerNorm) = FluxLayerNorm()
layertype(l::BatchNorm) = FluxBatchNorm()
layertype(l::InstanceNorm) = FluxInstanceNorm()
layertype(l::GroupNorm) = FluxGroupNorm()

# Transparent layers, i.e nin == nout always and there are no parameters
struct FluxNoParLayer <: FluxTransparentLayer end
layertype(l::MaxPool) = FluxNoParLayer()
layertype(l::MeanPool) = FluxNoParLayer()
layertype(l::Dropout) = FluxNoParLayer()
layertype(l::AlphaDropout) = FluxNoParLayer()

# Compositions? Might not have any common methods...
# MaxOut, Chain?
