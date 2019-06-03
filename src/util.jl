

NaiveNASlib.nin(l) = nin(layertype(l), l)
NaiveNASlib.nout(l) = nout(layertype(l), l)

NaiveNASlib.nin(::ParLayer, l) = size(weights(l), indim(l))
NaiveNASlib.nout(::ParLayer, l) = size(weights(l), outdim(l))

NaiveNASlib.nin(::ParInvLayer, l) = nout(l)

NaiveNASlib.nout(::ParDiagonal, l) = length(weights(l))
NaiveNASlib.nout(::ParInvLayer, l::LayerNorm) = nout(l.diag)
NaiveNASlib.nout(::ParNorm, l) = length(l.β)

NaiveNASlib.nout(::ParRecurrent, l) = div(size(weights(l), outdim(l)), outscale(l))

outscale(l) = outscale(layertype(l))
outscale(::ParRnn) = 1
outscale(::ParLstm) = 4
outscale(::ParGru) = 3

indim(l) = indim(layertype(l))
outdim(l) = outdim(layertype(l))

indim(::ParDense) = 2
outdim(::ParDense) = 1

indim(::ParRecurrent) = 2
outdim(::ParRecurrent) = 1

indim(::ParConv) = 4
outdim(::ParConv) = 3
indim(::ParConvVanilla) = 3
outdim(::ParConvVanilla) = 4

# Note: Contrary to other ML frameworks, bias seems to always be present in Flux
weights(l) = weights(layertype(l), l)
bias(l) = bias(layertype(l), l)

weights(::ParDense, l) = l.W.data
bias(::ParDense, l) = l.b.data

weights(::ParConv, l) = l.weight.data
bias(::ParConv, l) = l.bias.data

weights(::ParDiagonal, l) = l.α.data
bias(::ParDiagonal, l) = l.β.data

weights(::ParRecurrent, l) = l.cell.Wi.data
bias(::ParRecurrent, l) = l.cell.b.data

hiddenweights(l) = hiddenweights(layertype(l), l)
hiddenweights(::ParRecurrent, l) = l.cell.Wh.data
hiddenstate(l) = hiddenstate(layertype(l), l)
hiddenstate(::ParRecurrent, l) = Flux.hidden(l.cell).data
hiddenstate(::ParLstm, l) = [h.data for h in Flux.hidden(l.cell)]
state(l) = state(layertype(l), l)
state(::ParRecurrent, l) = l.state.data
state(::ParLstm, l) = [h.data for h in l.state]
