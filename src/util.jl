
NaiveNASlib.nin(l::ParLayer) = size(weights(l), indim(l))
NaiveNASlib.nout(l::ParLayer) = size(weights(l), outdim(l))

indim(l::ParDense) = 2
outdim(l::ParDense) = 1

indim(l::ParConv) = 3
outdim(l::ParConv) = 4

# Note: Contrary to other ML frameworks, bias seems to be always present in Flux
weights(l::Dense) = l.W.data
bias(l::Dense) = l.b.data

weights(l::ParConv) = l.weight.data
bias(l::ParConv) = l.bias.data
