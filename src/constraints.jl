
struct DepthWiseAllowNinChangeStrategy{S} <: NaiveNASlib.DecoratingJuMPΔSizeStrategy
  base::S
end
NaiveNASlib.base(s::DepthWiseAllowNinChangeStrategy) = s.base

struct DepthWiseSimpleΔSizeStrategy{S} <: NaiveNASlib.DecoratingJuMPΔSizeStrategy
  base::S
end
NaiveNASlib.base(s::DepthWiseSimpleΔSizeStrategy) = s.base


function NaiveNASlib.compconstraint!(case, s, ::FluxLayer, data) end
NaiveNASlib.compconstraint!(case, s::NaiveNASlib.DecoratingJuMPΔSizeStrategy, lt::FluxLayer, data) = NaiveNASlib.compconstraint!(case, NaiveNASlib.base(s), lt, data)
function NaiveNASlib.compconstraint!(::NaiveNASlib.ScalarSize, ::NaiveNASlib.AbstractJuMPΔSizeStrategy, ::FluxDepthwiseConv, data)

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

function NaiveNASlib.compconstraint!(case::NaiveNASlib.NeuronIndices, s::NaiveNASlib.AbstractJuMPΔSizeStrategy, t::FluxDepthwiseConv, data)
  if count(lt -> lt isa FluxDepthwiseConv, layertype.(keys(data.outselectvars))) > 2
    return NaiveNASlib.compconstraint!(case, DepthWiseSimpleΔSizeStrategy(s), t, data)
  end
  return NaiveNASlib.compconstraint!(case, DepthWiseAllowNinChangeStrategy(s), t, data)
end

function NaiveNASlib.compconstraint!(::NaiveNASlib.NeuronIndices, s::DepthWiseSimpleΔSizeStrategy, t::FluxDepthwiseConv, data)
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

  NaiveNASlib.compconstraint!(NaiveNASlib.ScalarSize(), base(s), t, data)
end

function NaiveNASlib.compconstraint!(case::NaiveNASlib.NeuronIndices, s::DepthWiseAllowNinChangeStrategy, t::FluxDepthwiseConv, data)
  model = data.model
  v = data.vertex
  select = data.outselectvars[v]
  insert = data.outinsertvars[v]

  nin(v)[] == 1 && return # Special case, no restrictions as we only need to be an integer multple of 1?

  # Step 1: 
  # Neurons mapped to the same weight are interleaved, i.e layer.weight[:,:,1,:] maps to y[1:ngroups:end] where y = layer(x)
  # where ngroups = nout / nin. For example, nout = 12 and nin = 4 mean size(layer.weight) == (..,3, 4)
  # Thus we have ngroups = 3 and with groupsize of 4.
  # Now, if we were to change nin to 3, we would still have ngroups = 3 but with groupsize 3 and thus nout = 9
  # 
  ins = filter(vin -> vin in keys(data.noutdict), inputs(v))
  # If inputs to v are not part of problem we have to keep nin(v) fixed!
  isempty(ins) && return NaiveNASlib.compconstraint!(case, DepthWiseSimpleΔSizeStrategy(base(s)), t, data)
  # TODO: Check if input is immutable and do simple strat then too?
  inselect = data.outselectvars[ins[]]
  ininsert = data.outinsertvars[ins[]]

  #ngroups = div(nout(v), nin(v)[])
  currgroupsize = nin(v)[]
  add_depthwise_constraints(model, inselect, ininsert, select, insert, currgroupsize)
end

function add_depthwise_constraints(model, inselect, ininsert, select, insert, currgroupsize)

  ngroups = div(length(select), currgroupsize)

  select_none_in_group = @variable(model, [1:ngroups], Bin)
  insert_none_in_group = @variable(model, [1:ngroups], Bin)

  insize = JuMP.@expression(model, sum(inselect) + sum(ininsert))

  # TODO: Redundant, or at least replace with constraint that all select_in_group[:, j] must be the same (unless select_none_in_group)?
  #   - No, I think it is correct. We have all selects mapped to an inselect except the dropped outgroups?
  @constraint(model, [i=1:length(select)], inselect[mod1(i, currgroupsize)] - select[i] + (select_none_in_group[mod1(i, ngroups)]) * 1e6 >= 0)
  @constraint(model, [i=1:length(select)], inselect[mod1(i, currgroupsize)] - select[i] - (select_none_in_group[mod1(i, ngroups)]) * 1e6 <= 0)

  insert_new_ingroups = @variable(model, [1, 1:currgroupsize], Int)
  @constraint(model, 0 .<= insert_new_ingroups .<= 100)

  @constraint(model,[i=1:length(ininsert)], ngroups * ininsert[i] == insert_new_ingroups[1, i])

  select_in_group = reshape(select, ngroups, currgroupsize)
  insert_in_group = reshape(insert, ngroups, currgroupsize)
  @constraint(model, sum(select_in_group, dims=2) .<= 1e6 .* (1 .- select_none_in_group))
  @constraint(model, sum(insert_in_group, dims=2) .<= 1e6 .* (1 .- insert_none_in_group))

  insert_new_groups = @variable(model, [1:ngroups, 1:currgroupsize], Int)
  @constraint(model, 0 .<= insert_new_groups .<= 100)
 
  @constraint(model,[g=1:ngroups, i=2:currgroupsize], insert_new_groups[g,i] == insert_new_groups[g,i-1])

  @constraint(model, [g=1:ngroups], insert_new_groups[g,1] == insert_new_groups[g,end])

  maxmultiplier = 10
  multipliers = @variable(model, [1:ngroups, 1:maxmultiplier], Bin)
  @constraint(model,[g=1:ngroups], sum(multipliers[g,:]) == length(multipliers[g,:]) - 1)

  @constraint(model, [g=1:ngroups, j=1:maxmultiplier], sum(insert_new_groups[g,:]) - (j-1)*insize + multipliers[g,j] * 1e6 >= 0)
  @constraint(model, [g=1:ngroups, j=1:maxmultiplier], sum(insert_new_groups[g,:]) - (j-1)*insize - multipliers[g,j] * 1e6 <= 0)

  @constraint(model, insert .== reshape(insert_new_groups, :) .+ reshape(vcat(zeros(ngroups-1,currgroupsize), insert_new_ingroups), :))
end

export testmod
function testmod(vals = [1, 2, 1, 2, 2, 3, 2, 3])
  model = NaiveNASlib.selectmodel(NaiveNASlib.NeuronIndices(), NaiveNASlib.DefaultJuMPΔSizeStrategy(), 1, 1)
  select = @variable(model, [1:length(vals)], Bin)
  insert = @variable(model, [1:length(vals)], Int)
  @constraint(model, 0 .<= insert .<= 100)

  ngroups = 3
  maxgroupsize = 10
  groupsizes = @variable(model, [1:maxgroupsize], Bin)
  select_none_in_group = @variable(model, [1:ngroups], Bin)

  invar = @variable(model, integer=true)

  @constraint(model, sum(groupsizes) == length(groupsizes)-1)
  for group in 1:ngroups
    select_in_group = select[group : ngroups : end]
    @constraint(model, sum(select_in_group) <= 1e6 * (1-select_none_in_group[group]))

    insert_in_group = insert[group : ngroups : end]
    for j in 1:length(groupsizes)
      jlim = min(j, length(insert_in_group))

      #@constraint(model, sum(select_in_group) + (groupsizes[j] + select_none_in_group[group]) * 1e6 >= jlim) 
      #@constraint(model, sum(select_in_group) - (groupsizes[j] + select_none_in_group[group]) * 1e6 <= jlim)    
      
      @constraint(model, [i=2:jlim], select_in_group[i] - select_in_group[i-1] - (groupsizes[j] + select_none_in_group[group]) * 1e6 <= 0)
      @constraint(model, [i=2:jlim], select_in_group[i] - select_in_group[i-1] + (groupsizes[j] + select_none_in_group[group]) * 1e6 >= 0)
      @constraint(model, select_in_group[1] - select_in_group[jlim] - (groupsizes[j] + select_none_in_group[group]) * 1e6 <= 0)
      @constraint(model, select_in_group[1] - select_in_group[jlim] + (groupsizes[j] + select_none_in_group[group])* 1e6 >= 0)

      @constraint(model, [i=j+1:length(select_in_group)], select_in_group[i] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, [i=j+1:length(select_in_group)], select_in_group[i] + groupsizes[j] * 1e6 >= 0)  

      @constraint(model, [i=2:jlim], insert_in_group[i] - insert_in_group[i-1] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, [i=2:jlim], insert_in_group[i] - insert_in_group[i-1] + groupsizes[j] * 1e6 >= 0)
      @constraint(model, insert_in_group[1] - insert_in_group[jlim] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, insert_in_group[1] - insert_in_group[jlim] + groupsizes[j] * 1e6 >= 0)

      @constraint(model, [i=j+1:length(insert_in_group)], insert_in_group[i] - groupsizes[j] * 1e6 <= 0)
      @constraint(model, [i=j+1:length(insert_in_group)], insert_in_group[i] + groupsizes[j] * 1e6 >= 0)  
    end
  end
  #@constraint(model, select .== 1)

  #@constraint(model, 12 == invar)

  inmultiplier = @variable(model, integer=true)
  @constraint(model, 0 <= inmultiplier <= 100)
  @constraint(model, [j=1:maxgroupsize], invar - j * inmultiplier + groupsizes[j] * 1e6 >= 0)
  @constraint(model, [j=1:maxgroupsize], invar - j * inmultiplier - groupsizes[j] * 1e6 <= 0)

  inmultipliers = @variable(model, [1:maxgroupsize], Bin)
  @constraint(model, sum(inmultipliers) == length(inmultipliers)-1)
  outgroups = JuMP.@expression(model, ngroups - sum(select_none_in_group))
  @constraint(model, [i=1:maxgroupsize, j=1:length(inmultipliers)], i * outgroups - j*invar + (inmultipliers[j] + groupsizes[i]) * 1e6 >= 0)
  @constraint(model, [i=1:maxgroupsize, j=1:length(inmultipliers)], i * outgroups - j*invar - (inmultipliers[j] + groupsizes[i]) * 1e6 <= 0)


  outvar = JuMP.@expression(model, sum(select) + sum(insert))
  multipliers = @variable(model, [1:maxgroupsize], Bin)
  @constraint(model, sum(multipliers) == length(multipliers)-1)
  @constraint(model, [j=1:length(multipliers)], outvar - j*invar + multipliers[j] * 1e6 >= 0)
  @constraint(model, [j=1:length(multipliers)], outvar - j*invar - multipliers[j] * 1e6 <= 0)


  JuMP.@objective(model, Max, sum(select .* vals) - 0.1sum(insert))

  #@constraint(model, sum(select) == 8)
  @constraint(model, sum(select) + sum(insert) == 8)

  JuMP.optimize!(model)

  if NaiveNASlib.accept(12,NaiveNASlib.DefaultJuMPΔSizeStrategy(), model)
    @show jval.(select)
    @show jval.(insert)
    @show sum(jval.(insert)) + sum(jval.(select))
    @show jval.(select_none_in_group)
    @show jval(invar)
    @show jval.(groupsizes)
    @show jval.(inmultiplier)
    @show jval.(inmultipliers)
    @show jval.(multipliers)
    @show NaiveNASlib.extract_inds(NaiveNASlib.DefaultJuMPΔSizeStrategy(), select, insert)
  end
  nothing
end
jval(var) = round(Int, JuMP.value(var))

export testmod2
function testmod2(vals, ngroups)

  model = NaiveNASlib.selectmodel(NaiveNASlib.NeuronIndices(), NaiveNASlib.DefaultJuMPΔSizeStrategy(), 1, 1)
  select = @variable(model, [1:length(vals)], Bin)
  insert = @variable(model, [1:length(vals)], Int)
  @constraint(model, 0 .<= insert .<= 100)

  currgroupsize = div(length(vals), ngroups)
  inselect = @variable(model, [1:currgroupsize], Bin)
  ininsert = @variable(model, [1:currgroupsize], Bin)

  select_none_in_group = @variable(model, [1:ngroups], Bin)
  insert_none_in_group = @variable(model, [1:ngroups], Bin)

  emptygroup = @variable(model, [1:ngroups], Bin)
  JuMP.@constraint(model, emptygroup .<= select_none_in_group)
  JuMP.@constraint(model, emptygroup .<= insert_none_in_group)

  insize = JuMP.@expression(model, sum(inselect) + sum(ininsert))
  @constraint(model, 1 <= insize <= 100)
  outsize = JuMP.@expression(model, sum(select) + sum(insert))

  # TODO: Redundant, or at least replace with constraint that all select_in_group[:, j] must be the same (unless select_none_in_group)?
  #   - No, I think it is correct. We have all selects mapped to an inselect except the dropped outgroups?
  @constraint(model, [i=1:length(select)], inselect[mod1(i, currgroupsize)] - select[i] + (select_none_in_group[mod1(i, ngroups)]) * 1e6 >= 0)
  @constraint(model, [i=1:length(select)], inselect[mod1(i, currgroupsize)] - select[i] - (select_none_in_group[mod1(i, ngroups)]) * 1e6 <= 0)

  insert_new_ingroups = @variable(model, [1, 1:currgroupsize], Int)
  @constraint(model, 0 .<= insert_new_ingroups .<= 100)

  @constraint(model,[i=1:length(ininsert)], ngroups * ininsert[i] == insert_new_ingroups[1, i])

  #@constraint(model, [i=1:length(ininsert)], ngroups * ininsert[i] - insert_new_ingroups[1, i] + no_ininsert[i] * 1e6 >= 0)
  #@constraint(model, [i=1:length(ininsert)], ngroups * ininsert[i] - insert_new_ingroups[1, i] - no_ininsert[i] * 1e6 <= 0)

  select_in_group = reshape(select, ngroups, currgroupsize)
  insert_in_group = reshape(insert, ngroups, currgroupsize)
  @constraint(model, sum(select_in_group, dims=2) .<= 1e6 .* (1 .- select_none_in_group))
  @constraint(model, sum(insert_in_group, dims=2) .<= 1e6 .* (1 .- insert_none_in_group))

  insert_new_groups = @variable(model, [1:ngroups, 1:currgroupsize], Int)
  @constraint(model, 0 .<= insert_new_groups .<= 100)
 
  @constraint(model,[g=1:ngroups, i=2:currgroupsize], insert_new_groups[g,i] == insert_new_groups[g,i-1])

  @constraint(model, [g=1:ngroups], insert_new_groups[g,1] == insert_new_groups[g,end])

  #@constraint(model,[g=1:ngroups], insert_new_groups[g,1] + (1-add_new_groups[g]) * 1e6 >= 1)
  #@constraint(model,[g=1:ngroups], insert_new_groups[g,1] - (1-add_new_groups[g]) * 1e6 <= 1e5)

  maxmultiplier = 10
  multipliers = @variable(model, [1:ngroups, 1:maxmultiplier], Bin)
  @constraint(model,[g=1:ngroups], sum(multipliers[g,:]) == length(multipliers[g,:]) - 1)
  #@constraint(model, [g=1:ngroups, j=1:maxmultiplier], sum(insert) - (j-1)*insize + multipliers[j] * 1e6 >= 0)
  #@constraint(model, [g=1:ngroups, j=1:maxmultiplier], sum(insert) - (j-1)*insize - multipliers[j] * 1e6 <= 0)
  @constraint(model, [g=1:ngroups, j=1:maxmultiplier], sum(insert_new_groups[g,:]) - (j-1)*insize + multipliers[g,j] * 1e6 >= 0)
  @constraint(model, [g=1:ngroups, j=1:maxmultiplier], sum(insert_new_groups[g,:]) - (j-1)*insize - multipliers[g,j] * 1e6 <= 0)

  #@constraint(model, [g=1:ngroups, j=1:maxmultiplier], sum(insert_new_groups[g,:]) - (j-1)*sum(inselect) + multipliers[g,j] * 1e6 >= 0)
  #@constraint(model, [g=1:ngroups, j=1:maxmultiplier], sum(insert_new_groups[g,:]) - (j-1)*sum(inselect) - multipliers[g,j] * 1e6 <= 0)


  @constraint(model, insert .== reshape(insert_new_groups, :) .+ reshape(vcat(zeros(ngroups-1,currgroupsize), insert_new_ingroups), :))

  #@constraint(model, multipliers[end,2] == 0)
  #@constraint(model, insert_new_groups[end,:] .== 1)
  #@constraint(model, insert_new_groups .== 2)
  #@constraint(model, insert_new_groups .== 0)

  
  for g in 1:ngroups
    #@constraint(model, sum(select_in_group[g,:]) + sum(insert_in_group[g,:]) - insize + (emptygroup[g] + add_new_groups) * 1e6 >= 0)
    #@constraint(model, sum(select_in_group[g,:]) + sum(insert_in_group[g,:]) - insize - (emptygroup[g] + add_new_groups) * 1e6 <= 0)

    #@constraint(model, sum(select_in_group[g,:]) - sum(inselect) + (select_none_in_group[g]) * 1e6 >= 0)
    #@constraint(model, sum(select_in_group[g,:]) - sum(inselect) - (select_none_in_group[g]) * 1e6 <= 0)
    #@constraint(model, sum(select_in_group[g,:]) - insize + (select_none_in_group[g] + 1 - add_new_groups[g]) * 1e6 >= 0)
    #@constraint(model, sum(select_in_group[g,:]) - insize - (select_none_in_group[g] + 1 - add_new_groups[g]) * 1e6 <= 0)


    #@constraint(model, insert_in_group[g,:] .- ininsert .+ (emptygroup[g]) .* 1e6 .>= 0)
    #@constraint(model, insert_in_group[g,:] .- ininsert .- (emptygroup[g]) .* 1e6 .<= 0)
  end
  #@constraint(model, insert_new_groups[end,:] .== 1)
  #@constraint(model, insert_new_groups[end,:] .== 1)
  
  @constraint(model, outsize == 16)
  #@constraint(model, inselect[[1,3]] .== 1)
  #@constraint(model, sum(inselect) == 4)
  #@constraint(model, insize == 4)
  #@constraint(model, ininsert[end] == 1)
  JuMP.@objective(model, Max, sum(select .* vals) - 0.1sum(insert) + sum(inselect) - 0.1sum(ininsert))
  JuMP.optimize!(model)

  if NaiveNASlib.accept(12,NaiveNASlib.DefaultJuMPΔSizeStrategy(), model)
    @show jval.(select)
    @show jval.(insert)
    @show jval.(inselect)
    @show jval.(ininsert)
    #@show jval.(ninsert)
    #@show sum(jval.(insert)) + sum(jval.(select))
    #@show jval.(groupsizes)
    @show jval.(select_none_in_group)
    @show jval.(insert_none_in_group)
    @show jval.(emptygroup)
    #@show jval.(add_new_groups)
    @show jval.(insert_new_groups)
    @show jval.(insert_new_ingroups[1,:]')
    @show jval(outsize)
    @show jval(insize)
    for g in 1:ngroups
      @show g
      @show jval.(select[g:ngroups:end])
      @show jval.(insert[g:ngroups:end])
      @show jval.(multipliers[g, :])
    end

    @show NaiveNASlib.extract_inds(NaiveNASlib.DefaultJuMPΔSizeStrategy(), select, insert)
    @show NaiveNASlib.extract_inds(NaiveNASlib.DefaultJuMPΔSizeStrategy(), inselect, ininsert)
  end
  nothing
end

export testmod3
function testmod3(vals, ngroups)
  model = NaiveNASlib.selectmodel(NaiveNASlib.NeuronIndices(), NaiveNASlib.DefaultJuMPΔSizeStrategy(), 1, 1)
  select = @variable(model, [1:length(vals)], Bin)
  insert = @variable(model, [1:length(vals)], Int)
  @constraint(model, 0 .<= insert .<= 100)

  currgroupsize = div(length(vals), ngroups)
  inselect = @variable(model, [1:currgroupsize], Bin)
  ininsert = @variable(model, [1:currgroupsize], Bin)

  add_depthwise_constraints(model, inselect, ininsert, select, insert, currgroupsize, vals)
  

  @constraint(model, sum(insert) + sum(select) == 8)
  #@constraint(model, inselect[[1,3]] .== 1)
  #@constraint(model, sum(inselect) == 4)
  @constraint(model, insize == 4)
  #@constraint(model, ininsert[end] == 1)
  JuMP.@objective(model, Max, sum(select .* vals) - 0.1sum(insert) + sum(inselect) - 0.1sum(ininsert))
  JuMP.optimize!(model)

  if NaiveNASlib.accept(12,NaiveNASlib.DefaultJuMPΔSizeStrategy(), model)
    @show jval.(select)
    @show jval.(insert)
    @show jval.(inselect)
    @show jval.(ininsert)
    #@show jval.(ninsert)
    #@show sum(jval.(insert)) + sum(jval.(select))
    #@show jval.(groupsizes)
    @show jval.(select_none_in_group)
    @show jval.(insert_none_in_group)
    @show jval.(add_new_groups)
    @show jval.(insert_new_groups)
    @show jval(insize)
    for g in 1:ngroups
      @show g
      @show jval.(select[g:ngroups:end])
      @show jval.(insert[g:ngroups:end])
    end
    @show jval.(multipliers)

    @show NaiveNASlib.extract_inds(NaiveNASlib.DefaultJuMPΔSizeStrategy(), select, insert)
    @show NaiveNASlib.extract_inds(NaiveNASlib.DefaultJuMPΔSizeStrategy(), inselect, ininsert)
  end
  nothing
end

