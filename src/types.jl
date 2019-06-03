
abstract type FluxLayer end

layertype(l::T) where T = @error "No FluxLayer type defined for $T"

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

abstract type FluxConvolutional <: FluxParLayer end
struct FluxConv <: FluxConvolutional end
struct FluxConvTranspose  <: FluxConvolutional end
struct FluxDepthwiseConv <: FluxConvolutional  end
layertype(l::Conv) = FluxConv()
layertype(l::ConvTranspose) = FluxConvTranspose()
layertype(l::DepthwiseConv) = FluxDepthwiseConv()

# Invariant layers with parameters, i.e nin == nout always and parameter selection must
# be performed
abstract type FluxParInvLayer <: FluxLayer end
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
struct FluxTransparentLayer <: FluxLayer end
layertype(l::MaxPool) = FluxTransparentLayer()
layertype(l::MeanPool) = FluxTransparentLayer()
layertype(l::Dropout) = FluxTransparentLayer()
layertype(l::AlphaDropout) = FluxTransparentLayer()

# Compositions? Might not have any common methods...
# MaxOut, Chain?
