
usesos1(model::JuMP.Model) = usesos1(JuMP.backend(model))
usesos1(m) = JuMP.MOI.supports_constraint(m, JuMP.MOI.VectorOfVariables, JuMP.MOI.SOS1)


"""
  
  GroupedConvAllowNinChangeStrategy(newoutputsmax::Integer, multipliersmax::Integer, base, [fallback])  
  GroupedConvAllowNinChangeStrategy(allowed_new_outgroups::AbstractVector{<:Integer}, allowed_multipliers::AbstractVector{<:Integer}, base, [fallback])
  
`DecoratingJuMPΔSizeStrategy` which allows both nin and nout of grouped `Conv` layers (i.e `Conv` with `groups` != 1) to change independently.

Might cause optimization to take very long time so use with care! Use [`GroupedConvSimpleΔSizeStrategy`](@ref)
if `GroupedConvAllowNinChangeStrategy` takes too long.

The elements of `allowed_new_outgroups` determine how many extra elements in the output dimension of the weight 
shall be tried for each existing output element. For example, for a `Conv((k1,k2), nin=>nout; groups=nin))` one 
must insert integer multiples of `nout / nin` elements at the time. With `nin/nout = k` and `allowed_new_outgroups = 0:3` it is allowed to insert 0, `k`, `2k` or `3k` new elements in the output dimension between each already existing element.

The elements of `allowed_multipliers` determine the total number of allowed output elements, i.e the allowed 
ratios of `nout / nin`.

If `fallback` is not provided, it will be derived from `base`.
"""
struct GroupedConvAllowNinChangeStrategy{S,F} <: DecoratingJuMPΔSizeStrategy
  allowed_new_outgroups::Vector{Int}
  allowed_multipliers::Vector{Int}
  base::S
  fallback::F
end
GroupedConvAllowNinChangeStrategy(newoutputsmax::Integer, multipliersmax::Integer,base,fb...) = GroupedConvAllowNinChangeStrategy(0:newoutputsmax, 1:multipliersmax, base, fb...)


function GroupedConvAllowNinChangeStrategy(
  allowed_new_outgroups::AbstractVector{<:Integer},
  allowed_multipliers::AbstractVector{<:Integer}, 
  base, fb= recurse_fallback(s -> GroupedConvAllowNinChangeStrategy(allowed_new_outgroups, allowed_multipliers, s), base)) 
  return GroupedConvAllowNinChangeStrategy(collect(Int, allowed_new_outgroups), collect(Int, allowed_multipliers), base, fb)
end


NaiveNASlib.base(s::GroupedConvAllowNinChangeStrategy) = s.base
NaiveNASlib.fallback(s::GroupedConvAllowNinChangeStrategy) = s.fallback

NaiveNASlib.add_participants!(s::GroupedConvAllowNinChangeStrategy, vs=AbstractVertex[]) = NaiveNASlib.add_participants!(base(s), vs)


"""
  GroupedConvSimpleΔSizeStrategy(base, [fallback])

`DecoratingJuMPΔSizeStrategy` which only allows nout of grouped `Conv` layers (i.e `Conv` with `groups` != 1) to change.

Use if [`GroupedConvAllowNinChangeStrategy`](@ref) takes too long to solve.

The elements of `allowed_multipliers` determine the total number of allowed output elements, i.e the allowed 
ratios of `nout / nin` (where `nin` is fixed).

If `fallback` is not provided, it will be derived from `base`.
"""
struct GroupedConvSimpleΔSizeStrategy{S, F} <: DecoratingJuMPΔSizeStrategy
  allowed_multipliers::Vector{Int}
  base::S
  fallback::F
end

GroupedConvSimpleΔSizeStrategy(maxms::Integer, base, fb...) = GroupedConvSimpleΔSizeStrategy(1:maxms, base, fb...)
function GroupedConvSimpleΔSizeStrategy(ms::AbstractVector{<:Integer}, base, fb=recurse_fallback(s -> GroupedConvSimpleΔSizeStrategy(ms, s), base)) 
  return GroupedConvSimpleΔSizeStrategy(collect(Int, ms), base, fb)
end
NaiveNASlib.base(s::GroupedConvSimpleΔSizeStrategy) = s.base
NaiveNASlib.fallback(s::GroupedConvSimpleΔSizeStrategy) = s.fallback

NaiveNASlib.add_participants!(s::GroupedConvSimpleΔSizeStrategy, vs=AbstractVertex[]) = NaiveNASlib.add_participants!(base(s), vs)


recurse_fallback(f, s::AbstractJuMPΔSizeStrategy) = wrap_fallback(f, NaiveNASlib.fallback(s))
recurse_fallback(f, s::NaiveNASlib.DefaultJuMPΔSizeStrategy) = s
recurse_fallback(f, s::NaiveNASlib.ThrowΔSizeFailError) = s
recurse_fallback(f, s::NaiveNASlib.ΔSizeFailNoOp) = s
recurse_fallback(f, s::NaiveNASlib.LogΔSizeExec) = NaiveNASlib.LogΔSizeExec(s.msgfun, s.level, f(s.andthen))

wrap_fallback(f, s) = f(s)
wrap_fallback(f, s::NaiveNASlib.LogΔSizeExec) = NaiveNASlib.LogΔSizeExec(s.msgfun, s.level, f(s.andthen))


function NaiveNASlib.compconstraint!(case, s, ::FluxLayer, data) end
function NaiveNASlib.compconstraint!(case, s::DecoratingJuMPΔSizeStrategy, lt::FluxLayer, data) 
   NaiveNASlib.compconstraint!(case, NaiveNASlib.base(s), lt, data)
end
# To avoid ambiguity
function NaiveNASlib.compconstraint!(case::NaiveNASlib.ScalarSize, s::DecoratingJuMPΔSizeStrategy, lt::FluxConvolutional, data)
  NaiveNASlib.compconstraint!(case, NaiveNASlib.base(s), lt, data)
end
function NaiveNASlib.compconstraint!(::NaiveNASlib.ScalarSize, s::AbstractJuMPΔSizeStrategy, ::FluxConvolutional, data, ms=allowed_multipliers(s))
  ngroups(data.vertex) == 1 && return

  # Add constraint that nout(l) == n * nin(l) where n is integer
  ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))

   # "data.noutdict[data.vertex] == data.noutdict[ins[i]] * x" where x is an integer variable is not linear
   # Instead we use the good old big-M strategy to set up a series of "or" constraints (exactly one of them must be true).
   # This is combined with the abs value formulation to force equality.
   # multiplier[j] is false if and only if data.noutdict[data.vertex] == ms[j]*data.noutdict[ins[i]]
   # all but one of multiplier must be true
   # Each constraint below is trivially true if multiplier[j] is true since 1e6 is way bigger than the difference between the two variables 
  multipliers = @variable(data.model, [1:length(ms)], Bin)
  @constraint(data.model, sum(multipliers) == length(multipliers)-1)
  @constraint(data.model, [i=1:length(ins),j=1:length(ms)], data.noutdict[data.vertex] - ms[j]*data.noutdict[ins[i]] + multipliers[j] * 1e6 >= 0)
  @constraint(data.model, [i=1:length(ins),j=1:length(ms)], data.noutdict[data.vertex] - ms[j]*data.noutdict[ins[i]] - multipliers[j] * 1e6 <= 0)

  # Inputs which does not have a variable, possibly because all_in_Δsize_graph did not consider it to be part of the set of vertices which may change
  # We will constrain data.vertex to have integer multiple of its current size
  fixedins = filter(vin -> vin ∉ ins, inputs(data.vertex))
  if !isempty(fixedins) 
    fv_out = @variable(data.model, integer=true)
    @constraint(data.model, [i=1:length(fixedins)], data.noutdict[data.vertex] == nout(fixedins[i]) * fv_out)
  end
end

allowed_multipliers(s::GroupedConvAllowNinChangeStrategy) = s.allowed_multipliers
allowed_multipliers(s::GroupedConvSimpleΔSizeStrategy) = s.allowed_multipliers
allowed_multipliers(::AbstractJuMPΔSizeStrategy) = 1:10


function NaiveNASlib.compconstraint!(case::NaiveNASlib.NeuronIndices, s::DecoratingJuMPΔSizeStrategy, t::FluxConvolutional, data) 
  NaiveNASlib.compconstraint!(case, base(s), t, data)
end
function NaiveNASlib.compconstraint!(case::NaiveNASlib.NeuronIndices, s::AbstractJuMPΔSizeStrategy, t::FluxConvolutional, data)
  ngroups(data.vertex) == 1 && return
  # Fallbacks don't matter here since we won't call it from below here, just add default so we don't accidentally crash due to some
  # strategy which hasn't defined a fallback
  if 15 < sum(keys(data.outselectvars)) do v
      ngroups(v) == 1 && return 0
      return log2(nout(v)) # Very roughly determined...
  end
    return NaiveNASlib.compconstraint!(case, GroupedConvSimpleΔSizeStrategy(10, s, NaiveNASlib.DefaultJuMPΔSizeStrategy()), t, data)
  end
  # The number of allowed multipliers can probably be better tuned, perhaps based on current size.
  return NaiveNASlib.compconstraint!(case, GroupedConvAllowNinChangeStrategy(10, 10, s, NaiveNASlib.DefaultJuMPΔSizeStrategy()), t, data)
  #=
  For benchmarking:
    using NaiveNASflux, Flux, NaiveNASlib.Advanced
    function timedwc(ds, ws)
      for w in ws
        for d in ds
          iv = conv2dinputvertex("in", w)
          dv = reduce((v, i) -> fluxvertex("dv$i", DepthwiseConv((1,1), nout(v) => fld1(i, 3) * nout(v)), v), 1:d; init=iv)   
          metric = sum(ancestors(dv)) do v
            layer(v) isa Conv || return 0
            return log2(nout(v))
          end
          res = @timed Δsize!(ΔNoutRelaxed(outputs(iv)[1] => w; fallback = ΔSizeFailNoOp()))
          println((width=w, depth = d, metric=metric, time=res.time))
          res.time < 5 || break
        end
      end
    end
  =#
end

function NaiveNASlib.compconstraint!(::NaiveNASlib.NeuronIndices, s::GroupedConvSimpleΔSizeStrategy, t::FluxConvolutional, data)
  model = data.model
  v = data.vertex
  select = data.outselectvars[v]
  insert = data.outinsertvars[v]


  ngroups(v) == 1 && return
  nin(v)[] == 1 && return # Special case, no restrictions as we only need to be an integer multple of 1

  if size(weights(layer(v)), indim(v)) != 1
    @warn "Handling of convolutional layers with groups != nin not implemented. Model might not be size aligned after mutation!"
  end

  # Neurons mapped to the same weight are interleaved, i.e layer.weight[:,:,1,:] maps to y[1:ngroups:end] where y = layer(x)
  ngrps = div(nout(v), nin(v)[])

  for group in 1:ngrps
    neurons_in_group = select[group : ngrps : end]
    @constraint(model, neurons_in_group[1] == neurons_in_group[end])
    @constraint(model, [i=2:length(neurons_in_group)], neurons_in_group[i] == neurons_in_group[i-1])

    insert_in_group = insert[group : ngrps : end]
    @constraint(model, insert_in_group[1] == insert_in_group[end])
    @constraint(model, [i=2:length(insert_in_group)], insert_in_group[i] == insert_in_group[i-1])
  end

  NaiveNASlib.compconstraint!(NaiveNASlib.ScalarSize(), s, t, data, allowed_multipliers(s))
end

function NaiveNASlib.compconstraint!(case::NaiveNASlib.NeuronIndices, s::GroupedConvAllowNinChangeStrategy, t::FluxConvolutional, data)
  model = data.model
  v = data.vertex
  select = data.outselectvars[v]
  insert = data.outinsertvars[v]

  ngroups(v) == 1 && return
  nin(v)[] == 1 && return # Special case, no restrictions as we only need to be an integer multple of 1?

  # Step 0:
  # Flux 0.13 changed the grouping of weigths so that size(layer.weight) = (..., nin / ngroups, nout)
  # We can get back the shape expected here through weightgroups = reshape(layer.weight, ..., nout / groups, nin)
  # Step 1: 
  # Neurons mapped to the same weight are interleaved, i.e layer.weight[:,:,1,:] maps to y[1:ngroups:end] where y = layer(x)
  # where ngroups = nout / nin. For example, nout = 12 and nin = 4 mean size(layer.weight) == (..,3, 4)
  # Thus we have ngroups = 3 and with groupsize of 4.
  # Now, if we were to change nin to 3, we would still have ngroups = 3 but with groupsize 3 and thus nout = 9
  # 
  ins = filter(vin -> vin in keys(data.noutdict), inputs(v))
  # If inputs to v are not part of problem we have to keep nin(v) fixed!
  isempty(ins) && return NaiveNASlib.compconstraint!(case, GroupedConvSimpleΔSizeStrategy(allowed_multipliers(s), base(s)), t, data)
  # TODO: Check if input is immutable and do simple strat then too?
  inselect = data.outselectvars[ins[]]
  ininsert = data.outinsertvars[ins[]]

  #ngroups = div(nout(v), nin(v)[])
  if size(weights(layer(v)), indim(v)) != 1
    @warn "Handling of convolutional layers with groups != nin not implemented. Model might not be size aligned after mutation!"
  end
  ningroups = nin(v)[]
  add_depthwise_constraints(model, inselect, ininsert, select, insert, ningroups, s.allowed_new_outgroups, s.allowed_multipliers)
end

function add_depthwise_constraints(model, inselect, ininsert, select, insert, ningroups, allowed_new_outgroups, allowed_multipliers)

  # ningroups is the number of "input elements" in the weight array, last dimension at the time of writing
  # noutgroups is the number of "output elements" in the weight array, second to last dimension at time of writing
  # nin is == ningroups while nout == ningroups * noutgroups

  # When changing nin by Δ, we will get ningroups += Δ which just means that we insert/remove Δ input elements.
  # Inserting one new input element at position i will get us noutgroups new consecutive outputputs at position i
  # Thus nout change by Δ * noutgroups.

  # Note: Flux 0.13 changed the grouping of weigths so that size(layer.weight) = (..., nin / ngroups, nout)
  # We can get back the shape expected here through weightgroups = reshape(layer.weight, ..., nout / groups, nin)
  # All examples below assume the pre-0.13 representation!

  # Example:

  # dc = DepthwiseConv((1,1), 3 => 9; bias=false);
  
  # size(dc.weight)
  # (1, 1, 3, 3)

  # indata = randn(Float32, 1,1,3, 1);

  # reshape(dc(indata),:)'
  # 1×9 adjoint(::Vector{Float32}) with eltype Float32:
  #  0.315234  0.267166  -0.227757  0.03523  0.0516438  -0.124007  -0.63409  0.0273361  -0.457146

  # Inserting zeros instead of adding/removing just as it is less verbose to do so.
  # dc.weight[:,:,:,2] .= 0;

  # reshape(dc(indata),:)'
  # 1×9 adjoint(::Vector{Float32}) with eltype Float32:
 #  0.315234  0.267166  -0.227757  0.0  0.0  0.0  -0.63409  0.0273361  -0.457146

  # When changing nout by Δ where Δ ==  k * ningroups where k is an integer we can simply just add/remove k output elements
  # Output elements are interleaved in the output of the layer, so if we add/remove output element i, the elements i:ningroups:end
  # will be added/dropped from the layer output.
  # When Δ is not an integer multiple things get pretty hairy and I'm not sure the constraints below handle all causes
  # We then basically need to change nin so that the new nout is an integer multiple of the new nin.

  # Example

  # dc = DepthwiseConv((1,1), 3 => 9; bias=false);

  # reshape(dc(indata),:)'
  # 1×9 adjoint(::Vector{Float32}) with eltype Float32:
  #  0.109549  0.273408  -0.299883  0.0131096  0.00240822  -0.0461159  -0.138605  -0.557134  0.0246111

  # julia> dc.weight[:,:,2,:] .= 0;

  # reshape(dc(indata),:)'
  # 1×9 adjoint(::Vector{Float32}) with eltype Float32:
  # 0.109549  0.0  -0.299883  0.0131096  0.0  -0.0461159  -0.138605  0.0  0.0246111

  # One thing which makes this a bit tricky is that while noutgruops and ningroups can change
  # we are here stuck with the select and insert arrays that correspond to the original sizes
  # As such, keep in mind that noutgroups and ningroups in this function always relate to the
  # lengths of the arrays of JuMP variables that we have right now.  

  M = 1e6
  noutgroups = div(length(select), ningroups)
  
  # select_none_in_group[g] == 1 if we drop group g
  # TODO: Replace by just sum(select_in_group[g,:]) below?
  select_none_in_group = @variable(model, [1:noutgroups], Bin)
  select_in_group = reshape(select, noutgroups, ningroups)
  @constraint(model, sum(select_in_group, dims=2) .<= M .* (1 .- select_none_in_group))

  # Tie selected input indices to selected output indices. If we are to drop any layer output indices, we need to drop the whole group
  # as one whole group is tied to a single element in the output element dimension of the weight array.
  @constraint(model, [g=1:noutgroups, i=1:ningroups], inselect[i] - select_in_group[g, i] + select_none_in_group[g] * M >= 0)
  @constraint(model, [g=1:noutgroups, i=1:ningroups], inselect[i] - select_in_group[g, i] - select_none_in_group[g] * M <= 0)

  # This variable handles the constraint that if we add a new input, we need to add noutgroups new outputs
  # Note that for now, it is only connected to the insert variable through the sum constraint below.
  insert_new_inoutgroups = @variable(model, [[1], 1:ningroups], Int, lower_bound=0, upper_bound=ningroups * maximum(allowed_multipliers))

  insize = @expression(model, sum(inselect) + sum(ininsert))
  outsize = @expression(model, sum(select) + sum(insert))

  # inmultipliers[j] == 1 if nout(v) == allowed_multipliers[j] * nin(v)[]
  inmultipliers = @variable(model, [1:length(allowed_multipliers)], Bin)
  if usesos1(model)
      #SOS1 == Only one can be non-zero. Not strictly needed, but it seems like it speeds up the solver
    @constraint(model, inmultipliers in SOS1(1:length(inmultipliers)))
  end
  @constraint(model, sum(inmultipliers) == 1)

  @constraint(model,[i=1:length(ininsert), j=1:length(inmultipliers)], allowed_multipliers[j] * ininsert[i] - insert_new_inoutgroups[1, i] + (1-inmultipliers[j]) * M >= 0)
  @constraint(model,[i=1:length(ininsert), j=1:length(inmultipliers)], allowed_multipliers[j] * ininsert[i] - insert_new_inoutgroups[1, i] - (1-inmultipliers[j]) * M <= 0)

  @constraint(model, [j=1:length(inmultipliers)], allowed_multipliers[j] * insize - outsize + (1-inmultipliers[j]) * M >= 0)
  @constraint(model, [j=1:length(inmultipliers)], allowed_multipliers[j] * insize - outsize - (1-inmultipliers[j]) * M <= 0)

  insert_new_inoutgroups_all_inds = vcat(zeros(noutgroups-1,ningroups), insert_new_inoutgroups)

  # This variable handles the constraint that if we want to increase the output size without changing 
  # the input size, we need to insert whole outgroups since all noutgroup outputs are tied to a single 
  # output element in the weight array.
  insert_new_outgroups = @variable(model, [1:noutgroups, 1:ningroups], Int, lower_bound = 0, upper_bound=noutgroups * maximum(allowed_new_outgroups))

  # insert_no_outgroup[g] == 1 if we don't insert a new output group after group g
  # TODO: Replace by just sum(insert_new_outgroups[g,:]) below?
  insert_no_outgroup = @variable(model, [1:noutgroups], Bin)
  @constraint(model, sum(insert_new_outgroups, dims=2) .<= M .* (1 .- insert_no_outgroup))

  # When adding a new output group, all inserts must be identical 
  # If we don't add any, all inserts are just 0
  @constraint(model, [g=1:noutgroups, i=2:ningroups], insert_new_outgroups[g,i] == insert_new_outgroups[g,i-1])
  @constraint(model, [g=1:noutgroups], insert_new_outgroups[g,1] == insert_new_outgroups[g,end])

  # new_outgroup[g,j] == 1 if we are inserting allowed_new_outgroups[j] new output groups after output group g
  noutmults = 1:length(allowed_new_outgroups)
  new_outgroup = @variable(model, [1:noutgroups, noutmults], Bin)
  if usesos1(model)
    #SOS1 == Only one can be non-zero. Not strictly needed, but it seems like it speeds up the solver
    @constraint(model,[g=1:noutgroups], new_outgroup[g,:] in SOS1(1:length(allowed_new_outgroups)))
  end
  @constraint(model,[g=1:noutgroups], sum(new_outgroup[g,:]) == 1)

  groupsum = @expression(model, [g=1:noutgroups], sum(insert_new_outgroups[g,:]) - sum(insert_new_inoutgroups_all_inds[g,:]))
  @constraint(model, [g=1:noutgroups, j=noutmults], groupsum[g] - allowed_new_outgroups[j]*insize + (1-new_outgroup[g,j] + insert_no_outgroup[g]) * M >= 0)
  @constraint(model, [g=1:noutgroups, j=noutmults], groupsum[g] - allowed_new_outgroups[j]*insize - (1-new_outgroup[g,j] + insert_no_outgroup[g]) * M <= 0)

  # Finally, we say what the inserts shall be: the sum of inserts from new inputs and new outputs
  # I think this allows for them to be somewhat independent, but there are still cases when they
  # can't change simultaneously. TODO: Check those cases
  @constraint(model, insert .== reshape(insert_new_outgroups, :) .+ reshape(insert_new_inoutgroups_all_inds,:))

end
