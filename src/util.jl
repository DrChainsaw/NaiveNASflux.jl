

NaiveNASlib.nin(l) = nin(layertype(l), l)
NaiveNASlib.nout(l) = nout(layertype(l), l)

NaiveNASlib.nin(::FluxParLayer, l) = size(weights(l), indim(l))
NaiveNASlib.nout(::FluxParLayer, l) = size(weights(l), outdim(l))

NaiveNASlib.nin(::FluxParInvLayer, l) = nout(l)

NaiveNASlib.nout(::FluxDiagonal, l) = length(weights(l))
NaiveNASlib.nout(::FluxParInvLayer, l::LayerNorm) = nout(l.diag)
NaiveNASlib.nout(::FluxParNorm, l) = length(l.β)

NaiveNASlib.nout(::FluxRecurrent, l) = div(size(weights(l), outdim(l)), outscale(l))

outscale(l) = outscale(layertype(l))
outscale(::FluxRnn) = 1
outscale(::FluxLstm) = 4
outscale(::FluxGru) = 3

indim(l) = indim(layertype(l))
outdim(l) = outdim(layertype(l))

indim(::FluxDense) = 2
outdim(::FluxDense) = 1

indim(::FluxRecurrent) = 2
outdim(::FluxRecurrent) = 1

indim(::FluxConvolutional) = 4
outdim(::FluxConvolutional) = 3
indim(::FluxConv) = 3
outdim(::FluxConv) = 4

# Note: Contrary to other ML frameworks, bias seems to always be present in Flux
weights(l) = weights(layertype(l), l)
bias(l) = bias(layertype(l), l)

weights(::FluxDense, l) = l.W.data
bias(::FluxDense, l) = l.b.data

weights(::FluxConvolutional, l) = l.weight.data
bias(::FluxConvolutional, l) = l.bias.data

weights(::FluxDiagonal, l) = l.α.data
bias(::FluxDiagonal, l) = l.β.data

weights(::FluxRecurrent, l) = l.cell.Wi.data
bias(::FluxRecurrent, l) = l.cell.b.data

hiddenweights(l) = hiddenweights(layertype(l), l)
hiddenweights(::FluxRecurrent, l) = l.cell.Wh.data
hiddenstate(l) = hiddenstate(layertype(l), l)
hiddenstate(::FluxRecurrent, l) = Flux.hidden(l.cell).data
hiddenstate(::FluxLstm, l) = [h.data for h in Flux.hidden(l.cell)]
state(l) = state(layertype(l), l)
state(::FluxRecurrent, l) = l.state.data
state(::FluxLstm, l) = [h.data for h in l.state]
