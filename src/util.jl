

NaiveNASlib.nin(l) = nin(layertype(l), l)
NaiveNASlib.nout(l) = nout(layertype(l), l)

NaiveNASlib.nin(t::FluxLayer, l) = throw(ArgumentError("Not implemented for $t"))
NaiveNASlib.nout(t::FluxLayer, l) = throw(ArgumentError("Not implemented for $t"))

NaiveNASlib.nin(::FluxParLayer, l) = size(weights(l), indim(l))
NaiveNASlib.nout(::FluxParLayer, l) = size(weights(l), outdim(l))
NaiveNASlib.nout(::FluxDepthwiseConv, l) = size(weights(l), outdim(l)) * nin(l)


NaiveNASlib.nin(::FluxParInvLayer, l) = nout(l)

NaiveNASlib.nout(::FluxDiagonal, l) = length(weights(l))
NaiveNASlib.nout(::FluxParInvLayer, l::LayerNorm) = nout(l.diag)
NaiveNASlib.nout(::FluxParNorm, l) = l.chs

NaiveNASlib.nout(::FluxRecurrent, l) = div(size(weights(l), outdim(l)), outscale(l))

NaiveNASlib.minΔninfactor(::FluxLayer, l) = 1
NaiveNASlib.minΔnoutfactor(::FluxLayer, l) = 1

outscale(l) = outscale(layertype(l))
outscale(::FluxRnn) = 1
outscale(::FluxLstm) = 4
outscale(::FluxGru) = 3

indim(l) = indim(layertype(l))
outdim(l) = outdim(layertype(l))
actdim(l) = actdim(layertype(l))
actrank(l) = actrank(layertype(l))

indim(t::FluxLayer) = throw(ArgumentError("Not implemented for $t"))
outdim(t::FluxLayer) = throw(ArgumentError("Not implemented for $t"))
actdim(t::FluxLayer) = throw(ArgumentError("Not implemented for $t"))
actrank(t::FluxLayer) = throw(ArgumentError("Not implemented for $t"))

indim(::FluxDense) = 2
outdim(::FluxDense) = 1
actdim(::FluxDense) = 1
actrank(::FluxDense) = 1

indim(::FluxDiagonal) = 1
outdim(::FluxDiagonal) = 1
actdim(::FluxDiagonal) = 1
actrank(::FluxDiagonal) = 1

indim(::FluxRecurrent) = 2
outdim(::FluxRecurrent) = 1
actdim(::FluxRecurrent) = 1
actrank(::FluxRecurrent) = 2

indim(::FluxConvolutional{N}) where N = 2+N
outdim(::FluxConvolutional{N}) where N = 1+N
actdim(::FluxConvolutional{N}) where N = 1+N
actrank(::FluxConvolutional{N}) where N = 1+N
indim(::Union{FluxConv{N}, FluxCrossCor{N}}) where N = 1+N
outdim(::Union{FluxConv{N}, FluxCrossCor{N}}) where N = 2+N
# Note: Absence of bias mean that bias is of type Flux.Zeros which mostly behaves like a normal array, mostly...
weights(l) = weights(layertype(l), l)
bias(l) = bias(layertype(l), l)

weights(::FluxDense, l) = l.weight
bias(::FluxDense, l) = l.bias

weights(::FluxConvolutional, l) = l.weight
bias(::FluxConvolutional, l) = l.bias

weights(::FluxDiagonal, l) = l.α
bias(::FluxDiagonal, l) = l.β

weights(::FluxRecurrent, l) = l.cell.Wi
bias(::FluxRecurrent, l) = l.cell.b

hiddenweights(l) = hiddenweights(layertype(l), l)
hiddenweights(::FluxRecurrent, l) = l.cell.Wh
hiddenstate(l) = hiddenstate(layertype(l), l)
hiddenstate(::FluxRecurrent, l) = l.cell.state0
hiddenstate(::FluxLstm, l) = [h for h in l.cell.state0]
state(l) = state(layertype(l), l)
state(::FluxRecurrent, l) = l.state
state(::FluxLstm, l) = [h for h in l.state]


function NaiveNASlib.compconstraint!(case, s, ::FluxLayer, data) end
NaiveNASlib.compconstraint!(case, s::NaiveNASlib.DecoratingJuMPΔSizeStrategy, lt::FluxLayer, data) = NaiveNASlib.compconstraint!(case, NaiveNASlib.base(s), lt, data)
function NaiveNASlib.compconstraint!(case, ::NaiveNASlib.AbstractJuMPΔSizeStrategy, ::FluxDepthwiseConv, data)

  # Add constraint that nout(l) == n * nin(l) where n is integer
  ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))

   # "data.noutdict[data.vertex] == data.noutdict[ins[i]] * x" where x is an integer variable is not linear
   # Instead we use the good old big-M strategy to set up a series of "or" constraints (exactly one of them must be true).
   # This is combined with the abs value formulation to force equality.
   # multiplier[j] is false if and only if data.noutdict[data.vertex] == j*data.noutdict[ins[i]]
   # all but one of multiplier must be true
   # Each constraint below is trivially true if multiplier[j] is true since 1e6 is way bigger than the difference between the two variables 
  multipliers = @variable(data.model, [1:10], Bin)
  @constraint(data.model, sum(multipliers) == length(multipliers)-1)
  @constraint(data.model, [i=1:length(ins),j=1:length(multipliers)], data.noutdict[data.vertex] - j*data.noutdict[ins[i]] + multipliers[j] * 1e6 >= 0)
  @constraint(data.model, [i=1:length(ins),j=1:length(multipliers)], data.noutdict[data.vertex] - j*data.noutdict[ins[i]] - multipliers[j] * 1e6 <= 0)

  # Inputs which does not have a variable, possibly because all_in_Δsize_graph did not consider it to be part of the set of vertices which may change
  # We will constrain data.vertex to have integer multiple of its current size
  fixedins = filter(vin -> vin ∉ ins, inputs(data.vertex))
  if !isempty(fixedins) 
    fv_out = @variable(data.model, integer=true)
    @constraint(data.model, [i=1:length(fixedins)], data.noutdict[data.vertex] == nout(fixedins[i]) * fv_out)
  end
end

function NaiveNASlib.compconstraint!(::NaiveNASlib.NeuronIndices, s::NaiveNASlib.AbstractJuMPΔSizeStrategy, t::FluxDepthwiseConv, data)
  model = data.model
  v = data.vertex
  select = data.outselectvars[v]

  groupsize = div(nout(v), div(nout(v), nin(v)[]))
  for neurons_in_group in Iterators.partition(select, groupsize)
    @constraint(model, neurons_in_group[1] == neurons_in_group[end])
    @constraint(model, [i=2:groupsize], neurons_in_group[i] >= neurons_in_group[i-1])
  end

  NaiveNASlib.compconstraint!(NaiveNASlib.ScalarSize(), s, t, data)
end