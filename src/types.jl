
abstract type Layer end

layertype(l::T) where T = @error "No layer type defined for $T"

# Types of layers parameterized with weights and biases and with similar handling
# w.r.t what weights and biases means in terms of number of inputs and number of outputs
abstract type ParLayer <: Layer end
struct ParDense <: ParLayer end
struct ParRnn <:ParLayer end

abstract type ParConv <: ParLayer end
struct ParConvVanilla <: ParConv end
struct ParConvTranspose  <: ParConv end
struct ParDepthwiseConv <: ParConv  end

layertype(l::Dense) = ParDense()
layertype(l::Flux.Recur) = ParRnn()

layertype(l::Conv) = ParConvVanilla()
layertype(l::ConvTranspose) = ParConvTranspose()
layertype(l::DepthwiseConv) = ParDepthwiseConv()

# Invariant layers with parameters, i.e nin == nout always and parameter selection must
# be performed
abstract type ParInvLayer <: Layer end
struct ParDiagonal <: ParInvLayer end
struct ParLayerNorm <: ParInvLayer end
abstract type  ParNorm <: ParInvLayer end
struct ParBatchNorm <: ParNorm end
struct ParInstanceNorm <: ParNorm end
struct ParGroupNorm <: ParNorm end

layertype(l::Flux.Diagonal) = ParDiagonal()
layertype(l::LayerNorm) = ParLayerNorm()
layertype(l::BatchNorm) = ParBatchNorm()
layertype(l::InstanceNorm) = ParInstanceNorm()
layertype(l::GroupNorm) = ParGroupNorm()

# Transparent layers, i.e nin == nout always and there are no parameters
struct TransparentLayer <: Layer end
layertype(l::MaxPool) = TransparentLayer()
layertype(l::MeanPool) = TransparentLayer()
layertype(l::Dropout) = TransparentLayer()
layertype(l::AlphaDropout) = TransparentLayer()

# Compositions? Might not have any common methods...
# MaxOut, Chain?
