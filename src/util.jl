
NaiveNASlib.nin(l::ParLayer) = size(weights(l), indim(l))
NaiveNASlib.nout(l::ParLayer) = size(weights(l), outdim(l))

indim(l::ParDense) = 2
outdim(l::ParDense) = 1

weights(l::Dense) = l.W.data
bias(l::Dense) = l.b.data
