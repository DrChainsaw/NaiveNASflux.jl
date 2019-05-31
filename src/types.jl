
abstract type Layer end

layertype(l::T) where T = @error "No layer type defined for $T"

# Types of layers parameterized with weights and biases and with similar handling
# w.r.t what weights and biases means in terms of number of inputs and number of outputs
abstract type ParLayer <: Layer end
struct ParDense <: ParLayer end
struct ParRnn <:ParLayer end
struct ParConv <: ParLayer end

layertype(l::Dense) = ParDense()
layertype(l::Flux.Recur) = ParRnn()

layertype(l::Conv) = ParConv()
layertype(l::ConvTranspose) = ParConv()
layertype(l::DepthwiseConv) = ParConv()

# Invariant layers with parameters, i.e nin == nout always and parameter selection must
# be performed
struct ParInvLayer <: Layer end
layertype(l::Flux.Diagonal) = ParInvLayer()
layertype(l::LayerNorm) = ParInvLayer()
layertype(l::BatchNorm) = ParInvLayer()
layertype(l::InstanceNorm) = ParInvLayer()
layertype(l::GroupNorm) = ParInvLayer()

# Transparent layers, i.e nin == nout always and there are no parameters
struct TransparentLayer <: Layer end
layertype(l::MaxPool) = TransparentLayer()
layertype(l::MeanPool) = TransparentLayer()
layertype(l::Dropout) = TransparentLayer()
layertype(l::AlphaDropout) = TransparentLayer()

# Compositions? Might not have any common methods...
# MaxOut, Chain?
