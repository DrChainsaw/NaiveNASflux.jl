

NaiveNASlib.nin(l) = nin(layertype(l), l)
NaiveNASlib.nout(l) = nout(layertype(l), l)

NaiveNASlib.nin(::ParLayer, l) = size(weights(l), indim(l))
NaiveNASlib.nout(::ParLayer, l) = size(weights(l), outdim(l))

indim(l) = indim(layertype(l))
outdim(l) = outdim(layertype(l))

indim(::ParDense) = 2
outdim(::ParDense) = 1

indim(::ParConv) = 3
outdim(::ParConv) = 4

# Note: Contrary to other ML frameworks, bias seems to always be present in Flux
weights(l) = weights(layertype(l), l)
bias(l) = bias(layertype(l), l)

weights(::ParDense, l) = l.W.data
bias(::ParDense, l) = l.b.data

weights(::ParConv, l) = l.weight.data
bias(::ParConv, l) = l.bias.data
