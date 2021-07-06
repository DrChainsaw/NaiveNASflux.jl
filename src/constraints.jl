

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
  insert = data.outinsertvars[v]

  nin(v)[] == 1 && return # Special case, no restrictions as we only need to be an integer multple of 1

  ngroups = div(nout(v), nin(v)[])
  # Neurons mapped to the same weight are interleaved, i.e layer.weight[:,:,1,:] maps to y[1:ngroups:end] where y = layer(x)
  for group in 1:ngroups
    neurons_in_group = select[group : ngroups : end]
    @constraint(model, neurons_in_group[1] == neurons_in_group[end])
    @constraint(model, [i=2:length(neurons_in_group)], neurons_in_group[i] == neurons_in_group[i-1])

    insert_in_group = insert[group : ngroups : end]
    @constraint(model, insert_in_group[1] == insert_in_group[end])
    @constraint(model, [i=2:length(insert_in_group)], insert_in_group[i] == insert_in_group[i-1])
  end

  NaiveNASlib.compconstraint!(NaiveNASlib.ScalarSize(), s, t, data)
end